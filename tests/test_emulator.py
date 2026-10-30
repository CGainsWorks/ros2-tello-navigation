"""The H.264 path, the SDK emulator and the real driver against the emulator.

Three layers, each tested on its own before the next relies on it:

1. :mod:`tello_core.link.video`: libx264 encodes rendered frames, the stream is
   cut into datagrams and FFmpeg's parser and decoder put it back together,
   however it was cut and from wherever in the stream the decoder joins.
2. :mod:`tello_core.emulator`: the SDK as it is answered on the command port,
   the state broadcast, the dead-man landing and the outage schedule, spoken to
   through a bare UDP socket so nothing of the driver is involved.
3. :class:`~tello_core.link.udp.UdpLink` with :class:`H264Decoder` pointed at the
   emulator: connect, stream, decode, take off, land, exactly as it would be
   pointed at an aircraft.

Everything the emulator produces is simulated. None of this is evidence about a
DJI Tello; it is evidence that the driver, the decoder and the emulator agree.
"""

from __future__ import annotations

import socket
import time

import numpy as np
import pytest

av = pytest.importorskip("av", reason="PyAV is needed for the H.264 path")

from tello_core.emulator import EmulatorFaults, EmulatorSettings, TelloEmulator  # noqa: E402
from tello_core.link.state import decode_state  # noqa: E402
from tello_core.link.udp import UdpLink, UdpSettings  # noqa: E402
from tello_core.link.video import (  # noqa: E402
    TELLO_VIDEO_DATAGRAM,
    H264Decoder,
    H264Encoder,
    packetise,
)
from tello_core.localisation.qr import QrDetector, filter_known  # noqa: E402
from tello_core.types import Frame, RcCommand  # noqa: E402


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait(predicate, timeout: float, step: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return bool(predicate())


# ------------------------------------------------------------------ H.264 codec


@pytest.fixture(scope="module")
def rendered_frames(room, camera, config):
    """Eight frames along a short sideways glide, rendered without JPEG."""
    from tello_core.geometry import euler_to_matrix, make_transform
    from tello_core.sim.render import Renderer, RenderSettings

    renderer = Renderer(room, camera, RenderSettings(jpeg_quality=0, seed=3))
    frames = []
    for i in range(8):
        pose = (
            make_transform(euler_to_matrix(np.deg2rad(90.0)), np.array([1.0 + 0.02 * i, 0.9, 1.2]))
            @ config.mount.transform()
        )
        frames.append(renderer.render(pose)[:, :, 0])
    return frames


def _encode(frames, keyframe_interval=30):
    height, width = frames[0].shape
    encoder = H264Encoder(width, height, fps=30, keyframe_interval=keyframe_interval)
    stream = [encoder.encode(frame) for frame in frames]
    stream.append(encoder.close())
    return stream


def test_round_trip_through_tello_sized_datagrams(rendered_frames):
    stream = _encode(rendered_frames)
    decoder = H264Decoder()
    decoded = []
    for chunk in stream:
        for datagram in packetise(chunk):
            assert len(datagram) <= TELLO_VIDEO_DATAGRAM
            image = decoder(datagram)
            if image is not None:
                decoded.append(image)
    decoded.extend(decoder.flush())

    assert len(decoded) == len(rendered_frames)
    assert decoder.size == (960, 720)
    for original, image in zip(rendered_frames, decoded):
        assert image.shape == (720, 960, 3)
        assert image.dtype == np.uint8
        # Lossy, so close rather than equal. The bound is loose on purpose: it
        # catches a decoder producing the wrong frame or garbage, not a codec
        # setting that changes the artefacts.
        error = np.abs(image[:, :, 0].astype(int) - original.astype(int)).mean()
        assert error < 6.0


def test_each_frame_is_handed_over_on_its_own_last_datagram(rendered_frames):
    """No frame of parser latency when the stream uses the Tello framing."""
    stream = _encode(rendered_frames)
    decoder = H264Decoder()
    per_frame = []
    for chunk in stream[:-1]:
        datagrams = packetise(chunk)
        results = [decoder(d) for d in datagrams]
        per_frame.append(sum(r is not None for r in results))
        assert results[-1] is not None or len(chunk) % TELLO_VIDEO_DATAGRAM == 0
    assert per_frame == [1] * len(rendered_frames)


def test_decoder_does_not_care_how_the_stream_was_cut(rendered_frames):
    """The aircraft's datagram size is not documented, so with the end-of-frame
    hint switched off the decoder must not depend on it at all."""
    stream = b"".join(_encode(rendered_frames))
    decoder = H264Decoder(flush_on_short_datagram=False)
    count = sum(decoder(chunk) is not None for chunk in packetise(stream, 377))
    count += len(decoder.flush())
    assert count == len(rendered_frames)


def test_joining_mid_stream_waits_for_the_next_keyframe(rendered_frames):
    stream = _encode(rendered_frames, keyframe_interval=4)
    decoder = H264Decoder()
    decoded = []
    # Join after the first keyframe and its first P-frame have gone past.
    for chunk in stream[2:]:
        for datagram in packetise(chunk):
            image = decoder(datagram)
            if image is not None:
                decoded.append(image)
    decoded.extend(decoder.flush())
    # Frames 2 and 3 reference a keyframe the decoder never saw; the next
    # keyframe is frame 4 and everything from there decodes.
    assert len(decoded) == 4


def test_a_decoded_frame_still_localises(rendered_frames, marker_map):
    """The point of the whole path: markers survive H.264 and are decoded."""
    stream = b"".join(_encode(rendered_frames))
    decoder = H264Decoder()
    images = [image for image in map(decoder, packetise(stream)) if image is not None]
    images.extend(decoder.flush())
    observations = QrDetector().detect(Frame(timestamp=0.0, image=images[-1]))
    known, _ = filter_known(observations, marker_map)
    assert len(known) >= 1


# --------------------------------------------------------------- SDK emulator


class Client:
    """A bare UDP client standing where the driver would."""

    def __init__(self, emulator: TelloEmulator, state_port: int) -> None:
        self.address = emulator.command_address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.settimeout(0.3)
        self.state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state.bind(("127.0.0.1", state_port))
        self.state.settimeout(0.5)

    def send(self, text: str) -> None:
        self.socket.sendto(text.encode("ascii"), self.address)

    def ask(self, text: str, timeout: float = 0.3):
        self.send(text)
        return self.reply(timeout)

    def reply(self, timeout: float = 0.3):
        self.socket.settimeout(timeout)
        try:
            payload, sender = self.socket.recvfrom(4096)
        except socket.timeout:
            return None
        assert sender[1] == self.address[1], "replies come from the command port"
        return payload.decode("ascii")

    def close(self) -> None:
        self.socket.close()
        self.state.close()


@pytest.fixture
def emulator_factory(room, config, camera, tmp_path):
    made = []

    def make(start_yaw: float = 0.0, **overrides):
        state_port = _free_port()
        settings = EmulatorSettings(
            command_port=0,
            state_port=state_port,
            video_port=_free_port(),
            truth_path=str(tmp_path / f"truth{len(made)}.csv"),
            events_path=str(tmp_path / f"events{len(made)}.jsonl"),
            **overrides,
        )
        emulator = TelloEmulator(
            room,
            config,
            camera,
            settings,
            start_position=np.array([1.0, 0.9, 0.03]),
            start_yaw=start_yaw,
        ).start()
        client = Client(emulator, state_port)
        made.append((emulator, client))
        return emulator, client

    yield make
    for emulator, client in made:
        client.close()
        emulator.close()


def test_nothing_is_answered_before_sdk_mode(emulator_factory):
    _, client = emulator_factory()
    assert client.ask("battery?") is None
    assert client.ask("command") == "ok"
    assert client.ask("battery?") == "100"


def test_read_commands_and_errors(emulator_factory):
    _, client = emulator_factory(battery=0.62)
    assert client.ask("command") == "ok"
    assert client.ask("battery?") == "62"
    assert client.ask("height?") == "0dm"
    assert client.ask("speed?") == "100.0"
    assert client.ask("speed 5") == "error"
    assert client.ask("speed 50") == "ok"
    assert client.ask("speed?") == "50.0"
    assert client.ask("forward 100") == "error", "not flying"
    assert client.ask("land") == "error", "not flying"
    assert client.ask("flap") == "unknown command: flap"
    assert client.ask("wifi?").startswith("error")
    temp = client.ask("temp?")
    assert temp.endswith("C") and "~" in temp
    assert client.ask("attitude?").startswith("pitch:")


def test_state_is_broadcast_to_the_commanding_host(emulator_factory):
    _, client = emulator_factory(battery=0.87)
    assert client.ask("command") == "ok"
    payload, _ = client.state.recvfrom(4096)
    state = decode_state(payload.decode("ascii"), 0.0)
    assert state.valid
    assert state.battery == pytest.approx(0.87, abs=0.011)
    assert state.height < 0.05


def test_takeoff_is_answered_when_it_has_finished(emulator_factory):
    emulator, client = emulator_factory()
    assert client.ask("command") == "ok"
    sent = time.monotonic()
    assert client.ask("takeoff", timeout=0.2) is None, "no answer while still climbing"
    assert client.reply(timeout=5.0) == "ok"
    elapsed = time.monotonic() - sent
    assert elapsed > 1.0, "the climb takes time; an instant ok would be a lie"
    assert emulator.truth().position[2] > 0.9
    assert client.ask("takeoff") == "error", "already flying"
    height = client.ask("height?")
    assert height.endswith("dm") and int(height[:-2]) >= 9

    client.send("cw 90")
    yaw_before = emulator.truth().yaw
    assert client.reply(timeout=6.0) == "ok"
    turned = np.rad2deg(np.arctan2(np.sin(emulator.truth().yaw - yaw_before),
                                   np.cos(emulator.truth().yaw - yaw_before)))
    assert turned == pytest.approx(-90.0, abs=3.0), "cw is clockwise seen from above"

    client.send("land")
    assert client.reply(timeout=6.0) == "ok"
    assert not emulator.truth().flying


def test_rc_moves_the_aircraft_with_the_sdk_channel_signs(emulator_factory):
    emulator, client = emulator_factory()
    assert client.ask("command") == "ok"
    client.send("takeoff")
    assert client.reply(timeout=5.0) == "ok"
    start = emulator.truth().position.copy()
    # SDK channel a is positive to the right; the aircraft starts facing +x, so
    # right is -y in the map frame.
    for _ in range(15):
        client.send("rc 60 0 0 0")
        time.sleep(0.05)
    client.send("rc 0 0 0 0")
    moved = emulator.truth().position - start
    assert moved[1] < -0.1
    assert abs(moved[0]) < abs(moved[1])
    client.send("emergency")
    assert client.reply() == "ok"
    assert _wait(lambda: not emulator.truth().flying, 2.0)


def test_silence_lands_the_aircraft(emulator_factory):
    emulator, client = emulator_factory(command_timeout=1.0)
    assert client.ask("command") == "ok"
    client.send("takeoff")
    assert client.reply(timeout=5.0) == "ok"
    assert emulator.truth().flying
    assert _wait(lambda: not emulator.truth().flying, 8.0)
    assert emulator.stats()["auto_landed"]


def test_an_outage_silences_both_directions(emulator_factory):
    emulator, client = emulator_factory(faults=EmulatorFaults(outages=((2.5, 1.5),)))
    assert client.ask("command") == "ok"
    client.send("takeoff")
    assert client.reply(timeout=5.0) == "ok"
    assert _wait(lambda: not emulator.link_up(), 3.0)
    assert client.ask("battery?") is None, "commands do not arrive during an outage"
    assert emulator.counters["commands_dropped"] >= 1
    # Drain anything already queued, then confirm nothing new arrives.
    client.state.settimeout(0.05)
    try:
        while True:
            client.state.recvfrom(4096)
    except socket.timeout:
        pass
    if not emulator.link_up():
        client.state.settimeout(0.3)
        with pytest.raises(socket.timeout):
            client.state.recvfrom(4096)
    assert _wait(emulator.link_up, 3.0)
    assert client.ask("battery?") is not None


# ------------------------------------------------------- the driver, unchanged


def test_the_udp_driver_flies_the_emulator(emulator_factory, marker_map, config, camera):
    from tello_core.control.waypoint import heading_for_markers

    yaw = heading_for_markers(np.array([1.0, 0.9, 0.03]), marker_map, config, camera)
    emulator, client = emulator_factory(start_yaw=yaw)
    client.close()  # the driver binds the state port itself
    settings = UdpSettings(
        address=emulator.command_address,
        local_command_port=0,
        state_port=emulator.settings.state_port,
        video_port=emulator.settings.video_port,
    )
    decoder = H264Decoder()
    link = UdpLink(settings, video_decoder=decoder)
    try:
        assert link.connect(), link.status().detail
        assert link.start_video(), link.status().detail
        assert _wait(lambda: link.status().connected, 2.0)

        frames = []
        assert _wait(lambda: frames.append(link.poll_frame()) or any(frames), 10.0)
        frame = next(f for f in frames if f is not None)
        assert frame.image.shape == (720, 960, 3)
        known, _ = filter_known(QrDetector().detect(frame), marker_map)
        assert known, "the start pose faces the markers; the decoded frame must show them"

        states = []
        assert _wait(lambda: states.append(link.poll_state()) or any(states), 2.0)
        state = next(s for s in states if s is not None)
        assert state.valid and state.battery > 0.9

        link.takeoff()
        assert _wait(lambda: emulator.truth().position[2] > 0.9, 5.0)
        link.send_rc(RcCommand(left_right=0, forward_back=0, up_down=0, yaw_rate=0))
        link.land()
        assert _wait(lambda: not emulator.truth().flying, 6.0)
        assert link.status().packets_received > 10
    finally:
        link.close()


def test_pooled_rendering_streams_decodable_video(emulator_factory):
    """The render pool the emulator can use on a machine with cores to spare."""
    emulator, client = emulator_factory(render_workers=1, video_rate=10.0)
    video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    video.bind(("127.0.0.1", emulator.settings.video_port))
    video.settimeout(0.5)
    decoder = H264Decoder()
    try:
        assert client.ask("command") == "ok"
        assert client.ask("streamon") == "ok"
        frames = []
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and len(frames) < 3:
            try:
                payload, _ = video.recvfrom(2048)
            except socket.timeout:
                continue
            image = decoder(payload)
            if image is not None:
                frames.append(image)
        assert len(frames) >= 3
        assert frames[-1].shape == (720, 960, 3)
        assert emulator.stats()["render_delay_s"] > 0.0
    finally:
        video.close()
