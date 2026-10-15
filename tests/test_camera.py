"""The pinhole model and the two ways it is easy to misuse."""

from __future__ import annotations

import numpy as np
import pytest

from tello_core.camera import CameraModel


def test_nominal_model_matches_the_published_field_of_view(camera: CameraModel):
    """The shipped intrinsics are derived from an 82.6 degree diagonal.

    Worth a test rather than a comment: the derivation is the only justification
    for the focal length and the day someone edits the number without editing
    the reasoning is the day every position estimate acquires a scale error.
    """
    diagonal_fov = 2 * np.arctan(
        np.hypot(camera.width, camera.height) / (2 * camera.fx)
    )
    assert np.rad2deg(diagonal_fov) == pytest.approx(82.6, abs=0.1)
    assert camera.source == "nominal-fov"


def test_nominal_model_is_square_pixels_and_centred(camera: CameraModel):
    assert camera.fx == pytest.approx(camera.fy)
    assert camera.cx == pytest.approx((camera.width - 1) / 2)
    assert camera.cy == pytest.approx((camera.height - 1) / 2)


def test_field_of_view_properties(camera: CameraModel):
    assert 65 < np.rad2deg(camera.horizontal_fov) < 75
    assert 50 < np.rad2deg(camera.vertical_fov) < 60


def test_principal_point_projects_to_the_centre(camera: CameraModel):
    pixel = camera.project(np.array([[0.0, 0.0, 2.0]]))[0]
    assert pixel == pytest.approx([camera.cx, camera.cy])


def test_projection_scales_inversely_with_depth(camera: CameraModel):
    near = camera.project(np.array([[0.1, 0.0, 1.0]]))[0]
    far = camera.project(np.array([[0.1, 0.0, 2.0]]))[0]
    assert (near[0] - camera.cx) == pytest.approx(2 * (far[0] - camera.cx))


def test_points_behind_the_camera_project_to_nan(camera: CameraModel):
    """A marker behind the camera must not produce a plausible pixel.

    Dividing by a negative depth gives a finite coordinate that can easily land
    inside the image and a visibility test built on it will happily report a
    marker that is behind the aircraft. NaN propagates into the test instead.
    """
    pixels = camera.project(np.array([[0.1, 0.0, -1.0], [0.0, 0.0, 0.0]]))
    assert np.isnan(pixels).all()


def test_in_view_rejects_nan(camera: CameraModel):
    assert not camera.in_view(np.array([[np.nan, np.nan]])).any()


def test_in_view_respects_the_margin(camera: CameraModel):
    edge = np.array([[5.0, 5.0]])
    assert camera.in_view(edge, margin=0.0).all()
    assert not camera.in_view(edge, margin=10.0).any()


def test_scaled_camera_keeps_the_field_of_view(camera: CameraModel):
    """Halving the resolution must halve the focal length too.

    A scaled image with unscaled intrinsics is a silent factor-of-two error in
    every pose, which is why the only way to change resolution is this method.
    """
    half = camera.scaled(0.5)
    assert half.width == camera.width // 2
    assert half.horizontal_fov == pytest.approx(camera.horizontal_fov, rel=1e-3)
    assert half.vertical_fov == pytest.approx(camera.vertical_fov, rel=1e-3)


def test_scaled_camera_records_the_scale_in_its_source(camera: CameraModel):
    assert "scale0.5" in camera.scaled(0.5).source


def test_scaled_camera_rejects_a_non_positive_factor(camera: CameraModel):
    with pytest.raises(ValueError):
        camera.scaled(0.0)


def test_yaml_round_trip(tmp_path, camera: CameraModel):
    import yaml

    path = tmp_path / "camera.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(camera.to_dict(), handle)
    loaded = CameraModel.from_yaml(path)
    assert loaded.fx == pytest.approx(camera.fx)
    assert loaded.width == camera.width
    assert np.allclose(loaded.distortion, camera.distortion)


def test_distortion_is_applied_when_present(camera: CameraModel):
    distorted = CameraModel(
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        width=camera.width,
        height=camera.height,
        distortion=np.array([-0.2, 0.05, 0.0, 0.0, 0.0]),
        source="test",
    )
    point = np.array([[0.4, 0.3, 1.0]])
    assert not np.allclose(camera.project(point), distorted.project(point))
    # The principal point is a fixed point of any radial model.
    assert np.allclose(
        camera.project(np.array([[0.0, 0.0, 1.0]])),
        distorted.project(np.array([[0.0, 0.0, 1.0]])),
    )
