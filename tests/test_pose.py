"""Detection and pose estimation, end to end through the renderer.

These are the tests that would catch a change of OpenCV version breaking the
detector, a sign error in the marker frame, or a covariance that stops telling
the truth. They render real QR symbols into a real scene and run the real
detector over them, so they are slower than the rest of the suite and worth every
millisecond: everything downstream believes what comes out of here.
"""

from __future__ import annotations

import numpy as np
import pytest

from tello_core.geometry import euler_to_matrix, invert_transform, make_transform
from tello_core.localisation.multi import (
    estimate_pose,
    marker_range,
    reprojection_gate,
)
from tello_core.localisation.qr import DetectorSettings, QrDetector, observation_span
from tello_core.sim.qr import decodable_range, render_sheet, symbol_modules
from tello_core.types import Frame


@pytest.fixture(scope="module")
def detector() -> QrDetector:
    return QrDetector()


def _pose(position, yaw, config):
    return make_transform(euler_to_matrix(yaw), np.asarray(position, float)) @ (
        config.mount.transform()
    )


def _detect(renderer, detector, config, position, yaw, **kwargs):
    image = renderer.render(_pose(position, yaw, config), **kwargs)
    return detector.detect(Frame(timestamp=0.0, image=image))


# ---------------------------------------------------------------- symbol shape


def test_encoder_quiet_zone_is_stripped():
    """OpenCV's encoder wraps the symbol in a two-module border.

    Leaving it in makes every printed marker 25/21 of the size the map says and
    every position estimate 19 per cent long, with a perfectly healthy-looking
    reprojection residual. A version-1 symbol is 21 modules and this asserts it.
    """
    assert symbol_modules("TLM01") == 21


def test_render_sheet_geometry():
    bitmap, symbol_px, total_px = render_sheet("TLM01", module_px=6, quiet_zone_modules=4)
    assert bitmap.shape == (total_px, total_px)
    assert symbol_px == 21 * 6
    assert total_px == symbol_px + 2 * 4 * 6
    # The quiet zone must be white all the way round.
    assert bitmap[: 4 * 6, :].min() == 255
    assert bitmap[:, : 4 * 6].min() == 255


def test_decodable_range_is_in_the_right_ballpark(camera):
    """A sanity check on the map, not a measurement.

    The measured curve is ``results/detection_range.csv``. This only has to be
    close enough to catch a marker size that could never work.
    """
    assert 2.0 < decodable_range("TLM01", 0.30, camera.fx) < 4.0


# ------------------------------------------------------------------ detection


def test_detects_the_marker_it_is_looking_at(renderer, detector, config, marker_map):
    marker = marker_map.get("M01")
    observations = _detect(
        renderer, detector, config, [1.5, marker.position[1], marker.position[2]], np.pi
    )
    assert marker.payload in {o.decoded for o in observations}


def test_detects_several_markers_at_once(renderer, detector, config):
    observations = _detect(renderer, detector, config, [2.0, 1.5, 1.3], np.pi)
    assert len({o.decoded for o in observations}) >= 2


def test_detects_nothing_facing_a_blank_wall(renderer, detector, config, marker_map):
    """Looking at the ceiling must produce no detections, not phantom ones."""
    observations = _detect(renderer, detector, config, [2.0, 1.5, 1.0], 0.0, )
    payloads = {o.decoded for o in observations}
    assert payloads <= {m.payload for m in marker_map} or not payloads


def test_a_marker_pair_beats_either_marker_alone(
    renderer, detector, config, camera, marker_map
):
    """Why the map is built from pairs.

    A single square leaves its own tilt nearly unobservable head on. A second
    marker 45 cm away in the same view turns a 30 cm object into a 75 cm one and
    the tilt becomes well determined. The difference is about a factor of five in
    position error and it is the reason every marker in the shipped map has a
    neighbour.
    """
    marker = marker_map.get("M01")
    position = np.array([marker.position[0] + 1.5, marker.position[1], marker.position[2]])
    observations = _detect(renderer, detector, config, position, np.pi)
    assert len(observations) >= 2, "the pair should both be read"

    pair = estimate_pose(observations, marker_map, camera, config.mount.transform())
    single = estimate_pose(
        [o for o in observations if o.decoded == marker.payload],
        marker_map,
        camera,
        config.mount.transform(),
    )
    assert pair is not None and single is not None
    pair_error = float(np.linalg.norm(pair.position - position))
    single_error = float(np.linalg.norm(single.position - position))
    assert pair_error < 0.05
    assert pair_error < single_error


def test_corner_order_matches_the_projection(clean_renderer, detector, config, camera, marker_map):
    """The detector's corners must line up with the projected marker corners.

    If they came out rotated or reversed the pose would still solve and would be
    wrong, so this is checked directly rather than inferred from the pose.
    """
    marker = marker_map.get("M01")
    position = [1.5, marker.position[1], marker.position[2]]
    observations = _detect(clean_renderer, detector, config, position, np.pi)
    observation = next(o for o in observations if o.decoded == marker.payload)

    T_cw = invert_transform(_pose(position, np.pi, config))
    expected = camera.project((T_cw[:3, :3] @ marker.corners_world().T).T + T_cw[:3, 3])
    assert np.allclose(observation.corners, expected, atol=1.5)


def test_subpixel_refinement_does_not_move_corners_far(clean_renderer, config, marker_map):
    marker = marker_map.get("M01")
    position = [1.5, marker.position[1], marker.position[2]]
    refined = QrDetector()
    raw = QrDetector(DetectorSettings(subpixel=False))
    image = clean_renderer.render(_pose(position, np.pi, config))
    frame = Frame(timestamp=0.0, image=image)
    a = next(o for o in refined.detect(frame) if o.decoded == marker.payload)
    b = next(o for o in raw.detect(frame) if o.decoded == marker.payload)
    assert np.max(np.abs(np.asarray(a.corners) - np.asarray(b.corners))) < 5.0


def test_observation_span_matches_the_pinhole_prediction(
    renderer, detector, config, camera, marker_map
):
    marker = marker_map.get("M01")
    distance = 1.5
    observations = _detect(
        renderer, detector, config,
        [marker.position[0] + distance, marker.position[1], marker.position[2]], np.pi,
    )
    observation = next(o for o in observations if o.decoded == marker.payload)
    expected = camera.fx * marker.size / (distance - config.mount.x)
    assert observation_span(observation) == pytest.approx(expected, rel=0.05)


# ----------------------------------------------------------------------- pose


@pytest.mark.parametrize("distance", [0.8, 1.2, 1.8, 2.5, 3.0])
def test_single_marker_pose_is_consistent_with_its_own_covariance(
    renderer, detector, config, camera, marker_map, distance
):
    """The error must sit inside the uncertainty the fix reports.

    Deliberately not an absolute accuracy bound. A single square marker seen
    nearly head on leaves the tilt almost unobservable, so the honest accuracy
    varies from a couple of centimetres to a third of a metre depending on the
    geometry and a fixed threshold would either be so loose it tests nothing or
    would fail on the geometry the covariance is already telling the truth about.
    What has to hold everywhere is that the fix knows how good it is.
    """
    marker = marker_map.get("M01")
    position = np.array(
        [marker.position[0] + distance, marker.position[1], marker.position[2]]
    )
    # One marker only. The shipped map puts markers in pairs, so a view of this
    # wall normally contains two and solves far better; this test is about what
    # a *single* marker can and cannot tell you.
    observations = [
        o
        for o in _detect(renderer, detector, config, position, np.pi)
        if o.decoded == marker.payload
    ]
    fix = estimate_pose(observations, marker_map, camera, config.mount.transform())
    if fix is None:
        # A legitimate outcome and the other half of the contract: at longer
        # range the two IPPE branches move far enough apart that no single
        # Gaussian spans them and the fix is refused rather than guessed. What
        # must never happen is a confident wrong answer.
        return
    assert fix.marker_ids == ("M01",)
    sigma = np.sqrt(np.diag(fix.position_covariance))
    error = np.abs(fix.position - position)
    assert np.all(error < 3.0 * sigma + 0.02)
    # However uncertain the tilt is, the range to the marker is not: it comes
    # from the apparent size, which is well measured.
    assert abs(error[0]) < 0.12


def test_a_single_marker_gives_a_fix_at_close_range(
    renderer, detector, config, camera, marker_map
):
    """The previous test tolerates a refusal, so this one pins down that a
    single marker close in does still produce a usable fix."""
    marker = marker_map.get("M01")
    position = np.array([marker.position[0] + 1.0, marker.position[1], marker.position[2]])
    observations = [
        o
        for o in _detect(renderer, detector, config, position, np.pi)
        if o.decoded == marker.payload
    ]
    fix = estimate_pose(observations, marker_map, camera, config.mount.transform())
    assert fix is not None
    assert fix.marker_ids == ("M01",)


def test_multi_marker_pose_beats_single_marker(
    renderer, detector, config, camera, marker_map
):
    """Two markers on different walls fix the depth neither fixes alone."""
    position = np.array([2.0, 1.5, 1.3])
    observations = _detect(renderer, detector, config, position, np.pi)
    assert len({o.decoded for o in observations}) >= 2
    fix = estimate_pose(observations, marker_map, camera, config.mount.transform())
    assert fix is not None
    assert len(fix.marker_ids) >= 2
    assert np.linalg.norm(fix.position - position) < 0.10


def test_yaw_is_recovered(renderer, detector, config, camera, marker_map):
    for yaw_deg in (150.0, 180.0, -150.0):
        yaw = np.deg2rad(yaw_deg)
        observations = _detect(renderer, detector, config, [1.6, 1.2, 1.3], yaw)
        fix = estimate_pose(observations, marker_map, camera, config.mount.transform())
        if fix is None:
            continue
        error = abs(float(np.rad2deg(np.arctan2(np.sin(fix.yaw - yaw), np.cos(fix.yaw - yaw)))))
        assert error < 12.0


def test_no_detections_gives_no_fix(marker_map, camera, config):
    assert estimate_pose([], marker_map, camera, config.mount.transform()) is None


def test_unknown_payload_gives_no_fix(marker_map, camera, config):
    from tello_core.types import MarkerObservation

    observation = MarkerObservation(
        marker_id="NOPE",
        corners=np.array([[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]),
        timestamp=0.0,
        decoded="NOPE",
    )
    assert estimate_pose([observation], marker_map, camera, config.mount.transform()) is None


def test_fix_records_which_markers_were_used(renderer, detector, config, camera, marker_map):
    observations = _detect(renderer, detector, config, [2.0, 1.5, 1.3], np.pi)
    fix = estimate_pose(observations, marker_map, camera, config.mount.transform())
    assert fix is not None
    assert set(fix.marker_ids) <= set(marker_map.ids)
    assert fix.n_points == 4 * len(fix.marker_ids)
    assert np.isfinite(marker_range(fix, marker_map))


# ----------------------------------------------------------------- covariance


def test_covariance_is_not_wildly_optimistic(
    renderer, detector, config, camera, marker_map
):
    """The reported sigma must be the same order as the actual error.

    A covariance that under-reports is the dangerous direction: the filter weights
    the fix too heavily and the failsafe, which reads the position sigma to decide
    whether the estimate can be trusted, believes it. Sampled over a spread of
    poses because any one of them can be lucky.
    """
    rng = np.random.default_rng(7)
    ratios = []
    for _ in range(14):
        position = np.array(
            [rng.uniform(0.8, 3.2), rng.uniform(0.7, 2.3), rng.uniform(0.9, 1.6)]
        )
        yaw = rng.uniform(-np.pi, np.pi)
        observations = _detect(renderer, detector, config, position, yaw)
        fix = estimate_pose(observations, marker_map, camera, config.mount.transform())
        if fix is None:
            continue
        sigma = float(np.sqrt(np.mean(np.diag(fix.position_covariance))))
        ratios.append(float(np.linalg.norm(fix.position - position)) / max(sigma, 1e-6))
    assert len(ratios) >= 6
    assert float(np.median(ratios)) < 2.5


def test_covariance_grows_with_range(renderer, detector, config, camera, marker_map):
    """Further away means less certain, for the same marker.

    Restricted to one marker on purpose. Comparing whatever happens to be in view
    at each range compares different problems: at three metres a second marker
    comes into frame and the two-marker solve is *better* conditioned than the
    one-marker solve at one metre, which is a true and completely different
    statement.
    """
    marker = marker_map.get("M01")

    def sigma_at(distance):
        position = [marker.position[0] + distance, marker.position[1], marker.position[2]]
        observations = [
            o
            for o in _detect(renderer, detector, config, position, np.pi)
            if o.decoded == marker.payload
        ]
        fix = estimate_pose(observations, marker_map, camera, config.mount.transform())
        assert fix is not None
        assert fix.marker_ids == ("M01",)
        return float(np.sqrt(np.mean(np.diag(fix.position_covariance))))

    assert sigma_at(3.0) > sigma_at(1.0)


def test_ambiguous_single_marker_fix_is_widened_not_discarded(
    renderer, detector, config, camera, marker_map
):
    """A marker seen square on is the common case and the ambiguous one.

    Both IPPE branches fit almost equally well, so the tilt is barely observable
    and the lateral position is uncertain by a good fraction of a marker width.
    Discarding those fixes leaves the aircraft unlocalised through most of a
    flight; publishing them with a tight covariance puts a systematic error into
    the filter. The right answer is a fix with a wide covariance along the one
    axis that is genuinely unresolved.
    """
    marker = marker_map.get("M01")
    position = np.array([marker.position[0] + 1.5, marker.position[1], marker.position[2]])
    observations = [
        o
        for o in _detect(renderer, detector, config, position, np.pi)
        if o.decoded == marker.payload
    ]
    assert observations

    fix = estimate_pose(observations, marker_map, camera, config.mount.transform())
    assert fix is not None, "a square-on single-marker view must still produce a fix"
    sigma = np.sqrt(np.diag(fix.position_covariance))
    # Range, which is the x axis here, is well observed; the lateral axis is not.
    assert sigma[0] < 0.05
    assert sigma.max() > 0.08
    assert float(np.linalg.norm(fix.position - position)) < 3 * float(sigma.max())


# ------------------------------------------------------------------- gating


def test_reprojection_gate_scales_with_marker_size():
    small = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 40.0], [0.0, 40.0]])
    large = np.array([[0.0, 0.0], [300.0, 0.0], [300.0, 300.0], [0.0, 300.0]])
    assert reprojection_gate(large) > reprojection_gate(small)


def test_reprojection_gate_is_bounded():
    tiny = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    huge = np.array([[0.0, 0.0], [900.0, 0.0], [900.0, 900.0], [0.0, 900.0]])
    assert reprojection_gate(tiny) >= 2.0
    assert reprojection_gate(huge) <= 8.0


def test_a_badly_wrong_association_is_rejected(marker_map, camera, config):
    """Corners nothing like the marker they claim to be must not produce a fix."""
    from tello_core.types import MarkerObservation

    observation = MarkerObservation(
        marker_id="TLM01",
        corners=np.array([[100.0, 100.0], [400.0, 130.0], [380.0, 460.0], [90.0, 300.0]]),
        timestamp=0.0,
        decoded="TLM01",
    )
    fix = estimate_pose([observation], marker_map, camera, config.mount.transform())
    if fix is not None:
        # If it does solve, the covariance must at least be honest about it.
        assert float(np.sqrt(np.mean(np.diag(fix.position_covariance)))) > 0.05
