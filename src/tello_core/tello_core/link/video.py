"""H.264 for the Tello's video stream, through PyAV (FFmpeg's libavcodec).

The Tello streams raw H.264 on UDP port 11111 after ``streamon``: an Annex B
byte stream (start codes, no container) cut into datagrams. Nothing in the
standard library can turn that into pixels, so this module wraps FFmpeg's H.264
parser and decoder through PyAV, which is what a team shipping a Tello driver in
Python would reach for. It is also what the emulator in
:mod:`tello_core.emulator` encodes with, so the decoder is exercised against a
real encoder's output rather than against frames this repository made up.

Decoding
--------
:class:`H264Decoder` is the ``video_decoder`` callable that
:class:`~tello_core.link.udp.UdpLink` takes. Each datagram goes through FFmpeg's
H.264 *parser* first, which finds access-unit boundaries in an arbitrary byte
stream. That makes the decoder indifferent to how the stream was cut into
datagrams, which matters because the packet size the aircraft uses is widely
described (1460-byte datagrams, with a short one ending each frame) and not
documented by DJI.

A parser on its own costs a frame of latency: it only knows a frame has ended
when the next one starts. At the emulator's achieved frame rate that was 140 ms
on every marker fix, which the fusion filter cannot tell from a wrong position,
and the first launch tests lost localisation on it. So by default a datagram
shorter than :data:`TELLO_VIDEO_DATAGRAM` also flushes the parser, which hands
the frame over the moment its last datagram arrives. That relies on the widely
described framing; ``flush_on_short_datagram=False`` turns it off and the
decoder then depends on nothing but the byte stream. ``docs/HARDWARE.md`` lists
checking which is right for a real aircraft.

A decoder that joins the stream mid-way, which is every decoder on a real link,
discards everything until the first keyframe with its parameter sets. Those
datagrams are counted rather than treated as errors.

PyAV is imported lazily, so ``tello_core`` itself still imports without it. The
pure-Python half of the repository (the simulator, the estimator and the unit
tests) has no use for a video codec.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Iterator, List, Optional

import numpy as np

TELLO_VIDEO_DATAGRAM = 1460
"""Bytes per video datagram, as the Tello is widely described to send them.

Not from DJI's SDK document, which does not specify it. The emulator cuts its
stream to this size so the driver sees datagrams shaped like the aircraft's. The
decoder uses a shorter datagram as an end-of-frame hint and can be told not to."""


def _import_av():
    try:
        import av  # noqa: PLC0415 - optional dependency, imported on first use
    except ImportError as error:  # pragma: no cover - exercised only without PyAV
        raise RuntimeError(
            "PyAV is not installed. The Tello's video is raw H.264 and decoding it "
            "needs FFmpeg: pip install av"
        ) from error
    return av


class H264Decoder:
    """Datagrams in, BGR images out.

    Callable with one UDP payload at a time. Returns the newest frame that
    payload completed, or ``None`` while the parser is still accumulating.
    """

    def __init__(self, flush_on_short_datagram: bool = True) -> None:
        av = _import_av()
        self._av = av
        self._context = av.CodecContext.create("h264", "r")
        self.flush_on_short_datagram = flush_on_short_datagram
        self.datagrams = 0
        self.bytes_in = 0
        self.frames_decoded = 0
        self.packets_rejected = 0
        """Access units the decoder refused, typically everything before the
        first keyframe after joining the stream."""

    @property
    def size(self) -> Optional[tuple]:
        """``(width, height)`` once the stream's parameter sets have been seen."""
        if self._context.width and self._context.height:
            return int(self._context.width), int(self._context.height)
        return None

    def __call__(self, datagram: bytes) -> Optional[np.ndarray]:
        self.datagrams += 1
        self.bytes_in += len(datagram)
        newest = None
        for image in self._decode(datagram):
            newest = image
        if self.flush_on_short_datagram and len(datagram) < TELLO_VIDEO_DATAGRAM:
            for image in self._decode(None, drain_decoder=False):
                newest = image
        return newest

    def flush(self) -> List[np.ndarray]:
        """Drain the parser and the decoder at the end of a stream."""
        return list(self._decode(None))

    def _decode(self, data: Optional[bytes], drain_decoder: bool = True) -> Iterator[np.ndarray]:
        """Parse ``data`` (``None`` flushes the parser) and decode what it yields.

        ``drain_decoder`` also flushes the decoder itself, which ends the stream;
        a frame-boundary flush must not do that.
        """
        error_types = (self._av.error.FFmpegError, ValueError)
        try:
            packets = self._context.parse(data)
        except error_types:
            self.packets_rejected += 1
            return
        if data is None and drain_decoder:
            packets = list(packets) + [None]
        for packet in packets:
            try:
                frames = self._context.decode(packet)
            except error_types:
                self.packets_rejected += 1
                continue
            for frame in frames:
                self.frames_decoded += 1
                yield frame.to_ndarray(format="bgr24")


class H264Encoder:
    """Greyscale images in, an Annex B H.264 byte stream out.

    The emulator's half of the pair. libx264 with the low-latency settings a
    live stream uses: no B-frames, no lookahead, a keyframe at a fixed interval
    and the parameter sets repeated on every keyframe so that a receiver joining
    mid-stream can start decoding at the next one. The bitrate and keyframe
    interval are emulator settings, not figures measured on a Tello.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float = 30.0,
        bitrate: int = 1_500_000,
        keyframe_interval: int = 30,
    ) -> None:
        av = _import_av()
        if width % 2 or height % 2:
            raise ValueError("H.264 4:2:0 needs even image dimensions")
        self.width = int(width)
        self.height = int(height)
        context = av.CodecContext.create("libx264", "w")
        context.width = self.width
        context.height = self.height
        context.pix_fmt = "yuv420p"
        rate = Fraction(fps).limit_denominator(1000)
        context.time_base = 1 / rate
        context.framerate = rate
        context.bit_rate = int(bitrate)
        gop = max(1, int(keyframe_interval))
        context.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "x264-params": (
                f"keyint={gop}:min-keyint={gop}:scenecut=0:bframes=0:repeat-headers=1"
            ),
        }
        self._av = av
        self._context = context
        self._chroma = np.full((self.height // 2, self.width), 128, dtype=np.uint8)
        self._pts = 0
        self.frames_encoded = 0
        self.bytes_out = 0

    def encode(self, gray: np.ndarray) -> bytes:
        """Encode one ``(H, W)`` uint8 frame and return whatever the encoder emits."""
        gray = np.asarray(gray)
        if gray.ndim == 3:
            gray = gray[:, :, 0]
        if gray.shape != (self.height, self.width):
            raise ValueError(f"expected {(self.height, self.width)}, got {gray.shape}")
        # The renderer is monochrome, so the chroma planes are flat grey. Built
        # directly as planar 4:2:0 rather than converted, which would cost a
        # colour-space conversion per frame to produce the same bytes. Luma goes
        # into the 16 to 235 video range, which is what a decoder assumes when
        # the stream does not say otherwise; full-range luma would come back
        # out of the decoder with its contrast stretched.
        luma = (16.0 + gray.astype(np.float32) * (219.0 / 255.0) + 0.5).astype(np.uint8)
        planar = np.concatenate([luma, self._chroma], axis=0)
        frame = self._av.VideoFrame.from_ndarray(planar, format="yuv420p")
        frame.pts = self._pts
        self._pts += 1
        payload = b"".join(bytes(packet) for packet in self._context.encode(frame))
        self.frames_encoded += 1
        self.bytes_out += len(payload)
        return payload

    def close(self) -> bytes:
        """Flush the encoder."""
        payload = b"".join(bytes(packet) for packet in self._context.encode(None))
        self.bytes_out += len(payload)
        return payload


def packetise(stream: bytes, size: int = TELLO_VIDEO_DATAGRAM) -> List[bytes]:
    """Cut one encoded frame into datagrams of at most ``size`` bytes.

    A frame that is an exact multiple of ``size`` gets no short datagram at the
    end, which is a known ambiguity in the framing as it is described and one
    more reason the decoder relies on the H.264 parser instead.
    """
    if size <= 0:
        raise ValueError("datagram size must be positive")
    return [stream[i : i + size] for i in range(0, len(stream), size)]
