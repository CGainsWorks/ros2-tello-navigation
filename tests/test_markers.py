"""The marker map: format, validation and the geometry the whole system rests on."""

from __future__ import annotations

import numpy as np
import pytest

from tello_core.geometry import euler_to_matrix, make_transform
from tello_core.localisation.markers import (
    Marker,
    MarkerMap,
    MarkerMapError,
    marker_from_pose,
    markers_in_view,
    pixel_span,
)

BASE = {
    "frame_id": "map",
    "default_size_m": 0.3,
    "markers": [
        {"id": "A", "payload": "A", "position": [0.0, 1.0, 1.4], "normal_yaw_deg": 0.0},
        {"id": "B", "payload": "B", "position": [4.0, 1.0, 1.4], "normal_yaw_deg": 180.0},
    ],
}


def _map(**overrides) -> dict:
    data = {k: (v.copy() if isinstance(v, list) else v) for k, v in BASE.items()}
    data["markers"] = [dict(m) for m in BASE["markers"]]
    data.update(overrides)
    return data


# ---------------------------------------------------------------- shipped map


def test_shipped_map_loads(marker_map: MarkerMap):
    assert len(marker_map) == 18
    assert marker_map.frame_id == "map"


def test_shipped_map_is_honest_about_not_being_surveyed(marker_map: MarkerMap):
    """The survey method must say ``designed`` until a room is actually measured.

    This test exists to make the claim hard to change by accident. Flipping it to
    ``laser`` without measuring anything would make every results file assert a
    provenance it does not have and the results files carry the value through
    into ``docs/RESULTS.md``.
    """
    assert marker_map.survey.method == "designed"


def test_shipped_map_geometry_is_sane(marker_map: MarkerMap):
    assert marker_map.check_geometry() == []


def test_shipped_map_has_vertical_spread(marker_map: MarkerMap):
    """Markers all at one height leave z poorly observed from a single marker."""
    heights = marker_map.positions()[:, 2]
    assert heights.max() - heights.min() > 1.0


def test_every_marker_has_a_neighbour_it_can_be_seen_with(marker_map: MarkerMap):
    """The layout is built from pairs and the pairing is what makes it accurate.

    A single square marker leaves its own tilt nearly unobservable head on and
    the resulting fix is good to about 15 cm; a second marker in the same view
    takes it under 3 cm. Every marker here has one within a metre on the same
    wall, so any heading that reads one reads two.
    """
    positions = marker_map.positions()
    for index, marker in enumerate(marker_map):
        others = np.delete(positions, index, axis=0)
        nearest = float(np.min(np.linalg.norm(others - marker.position, axis=1)))
        assert nearest < 1.0, f"{marker.marker_id} has no marker within a metre"


def test_shipped_map_round_trips(marker_map: MarkerMap):
    rebuilt = MarkerMap.from_dict(marker_map.to_dict())
    assert rebuilt.ids == marker_map.ids
    for original, copy in zip(marker_map, rebuilt):
        assert np.allclose(original.position, copy.position)
        assert np.allclose(original.rotation, copy.rotation, atol=1e-9)


# ------------------------------------------------------------------ validation


def test_duplicate_ids_are_rejected():
    data = _map()
    data["markers"][1]["id"] = "A"
    with pytest.raises(MarkerMapError, match="duplicate"):
        MarkerMap.from_dict(data)


def test_duplicate_payloads_are_rejected():
    """Two markers that decode to the same string cannot be told apart in flight."""
    data = _map()
    data["markers"][1]["payload"] = "A"
    with pytest.raises(MarkerMapError, match="already used"):
        MarkerMap.from_dict(data)


def test_non_ascii_payloads_are_rejected():
    data = _map()
    data["markers"][0]["payload"] = "MØ1"
    with pytest.raises(MarkerMapError, match="ASCII"):
        MarkerMap.from_dict(data)


def test_missing_size_is_rejected():
    """Size scales every position estimate, so there is no safe default."""
    data = _map()
    del data["default_size_m"]
    with pytest.raises(MarkerMapError, match="size_m"):
        MarkerMap.from_dict(data)


def test_bad_position_is_rejected():
    data = _map()
    data["markers"][0]["position"] = [0.0, 1.0]
    with pytest.raises(MarkerMapError, match="position"):
        MarkerMap.from_dict(data)


def test_non_finite_position_is_rejected():
    data = _map()
    data["markers"][0]["position"] = [0.0, float("nan"), 1.4]
    with pytest.raises(MarkerMapError, match="non-finite"):
        MarkerMap.from_dict(data)


def test_empty_map_is_rejected():
    with pytest.raises(MarkerMapError, match="no markers"):
        MarkerMap.from_dict({"markers": []})


def test_coincident_markers_produce_a_warning():
    data = _map()
    data["markers"][1]["position"] = [0.01, 1.0, 1.4]
    warnings = MarkerMap.from_dict(data).check_geometry()
    assert any("apart" in w for w in warnings)


def test_coplanar_markers_produce_a_warning():
    data = _map()
    data["markers"].append(
        {"id": "C", "payload": "C", "position": [0.0, 2.0, 1.0], "normal_yaw_deg": 0.0}
    )
    data["markers"][1]["position"] = [0.0, 0.5, 1.9]
    warnings = MarkerMap.from_dict(data).check_geometry()
    assert any("plane" in w for w in warnings)


# -------------------------------------------------------------------- geometry


def test_marker_frame_is_right_handed():
    marker = Marker("A", "A", 0.3, np.array([0.0, 1.0, 1.4]), 0.0)
    R = marker.rotation
    assert np.allclose(np.cross(R[:, 0], R[:, 1]), R[:, 2], atol=1e-12)
    assert float(np.linalg.det(R)) == pytest.approx(1.0)


def test_marker_is_not_mirrored():
    """A marker on the x = 0 wall must read the right way round from inside.

    The frame is OpenCV's: x right as printed, y up, z out of the paper. Standing
    in the room looking at that wall, your right is world +y, so the marker's own
    x must be +y. The combination that looks natural, x right with y *down*, is
    not right-handed with z towards the viewer and using it renders every marker
    mirrored. A mirrored marker still decodes and still solves and gives a pose
    that is wrong in a way no residual reveals.
    """
    marker = Marker("A", "A", 0.3, np.array([0.0, 1.0, 1.4]), 0.0)
    assert np.allclose(marker.rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
    assert np.allclose(marker.rotation @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])
    assert np.allclose(marker.normal_world(), [1.0, 0.0, 0.0])


def test_corner_order_is_top_left_first():
    marker = Marker("A", "A", 0.3, np.array([0.0, 1.0, 1.4]), 0.0)
    corners = marker.corners_world()
    # This marker is on the x = 0 wall facing into the room, so a viewer inside
    # the room faces -x with up +z and their right hand points along +y. The
    # marker's own x axis is that same +y, so "left as printed" is -y.
    #
    # Top-left: highest and on the -y side.
    assert corners[0][2] == pytest.approx(1.55)
    assert corners[0][1] == pytest.approx(0.85)
    # Bottom-right: lowest, +y side.
    assert corners[2][2] == pytest.approx(1.25)
    assert corners[2][1] == pytest.approx(1.15)


def test_corners_are_a_square_of_the_stated_size():
    marker = Marker("A", "A", 0.3, np.array([1.0, 2.0, 1.1]), np.deg2rad(37.0), tilt=0.1)
    corners = marker.corners_world()
    edges = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    assert np.allclose(edges, 0.3, atol=1e-12)


def test_roll_rotates_in_the_marker_plane():
    upright = Marker("A", "A", 0.3, np.zeros(3), 0.0)
    rolled = Marker("A", "A", 0.3, np.zeros(3), 0.0, roll=np.deg2rad(90.0))
    assert np.allclose(upright.normal_world(), rolled.normal_world(), atol=1e-12)
    assert not np.allclose(upright.corners_world(), rolled.corners_world())


def test_marker_from_pose_round_trips():
    original = Marker("A", "A", 0.3, np.array([1.0, 2.0, 1.1]), 0.9, tilt=0.12, roll=-0.07)
    recovered = marker_from_pose("A", "A", 0.3, original.pose)
    assert np.allclose(recovered.position, original.position)
    assert np.allclose(recovered.rotation, original.rotation, atol=1e-9)


# ------------------------------------------------------------------ visibility


def _camera_pose(position, yaw, config):
    return make_transform(euler_to_matrix(yaw), np.asarray(position, float)) @ (
        config.mount.transform()
    )


def test_markers_in_view_finds_the_wall_being_faced(marker_map, config, camera):
    pose = _camera_pose([1.5, 0.9, 1.4], np.pi, config)
    visible = {m.marker_id for m in markers_in_view(marker_map, pose, camera)}
    assert "M01" in visible


def test_markers_in_view_excludes_markers_behind(marker_map, config, camera):
    pose = _camera_pose([1.5, 0.9, 1.4], 0.0, config)
    visible = {m.marker_id for m in markers_in_view(marker_map, pose, camera)}
    assert "M01" not in visible


def test_markers_in_view_respects_max_range(marker_map, config, camera):
    pose = _camera_pose([3.5, 0.9, 1.4], np.pi, config)
    assert markers_in_view(marker_map, pose, camera, max_range=1.0) == []


def test_markers_in_view_rejects_extreme_incidence(marker_map, config, camera):
    """A marker seen almost edge on is geometrically in frame and useless."""
    pose = _camera_pose([0.5, 2.4, 1.4], np.deg2rad(-100.0), config)
    for marker in markers_in_view(marker_map, pose, camera):
        to_camera = pose[:3, 3] - marker.position
        cosine = float(
            np.dot(marker.normal_world(), to_camera / np.linalg.norm(to_camera))
        )
        assert cosine > np.cos(np.deg2rad(70.0))


def test_pixel_span_falls_with_distance(marker_map, config, camera):
    marker = marker_map.get("M01")
    near = pixel_span(marker, _camera_pose([1.0, 0.9, 1.4], np.pi, config), camera)
    far = pixel_span(marker, _camera_pose([2.0, 0.9, 1.4], np.pi, config), camera)
    assert near > far
    assert near / far == pytest.approx(2.0, rel=0.1)


def test_pixel_span_matches_the_pinhole_prediction(marker_map, config, camera):
    marker = marker_map.get("M01")
    distance = 1.5
    pose = _camera_pose([marker.position[0] + distance, marker.position[1],
                         marker.position[2]], np.pi, config)
    expected = camera.fx * marker.size / (distance - config.mount.x)
    assert pixel_span(marker, pose, camera) == pytest.approx(expected, rel=0.02)
