"""Turning marker detections into an absolute pose, with an honest covariance.

Two solvers, chosen automatically.

**One marker.** ``cv2.solvePnPGeneric`` with ``SOLVEPNP_IPPE_SQUARE``, which is
the closed-form solution for a planar square and is both faster and better
conditioned than the iterative solver for this case. Because the marker's four
corners come out of the detector in a known order there is no four-fold rotation
ambiguity to resolve.

There is, however, the planar two-fold ambiguity and it is not a technicality.
IPPE returns *two* poses that are both consistent with a square seen through a
pinhole: the true one and its reflection in the plane through the camera centre.
When the marker is close and seen at an angle the reflection reprojects visibly
worse and picking the better one is safe. When it is far away and nearly square
on, the two reprojection errors are within a few per cent of each other, the
solver picks essentially at random and the loser is not slightly wrong: it is
metres wrong, with a small residual and a confident covariance. On this
project's own renderer, taking the best-fitting solution blindly produced a
2.3 m error at 3.4 m range with a 0.6 px residual, which no gate on residual
could ever have caught.

So both solutions are kept and resolved in that order of preference: fit first,
then a prior, then neither. If one branch reprojects at least ``AMBIGUITY_RATIO``
times better than the other, the fit has decided and nothing else gets a vote. If
it has not and a prior position is available, the prior may break the tie but
only when it genuinely separates the two: one branch inside three sigma of the
prior and the other outside.

When nothing decides, the fix is **not** thrown away. It is published with a
covariance widened along the line between the two branches by half their
separation, which is the right Gaussian approximation to a two-mode posterior.
Throwing it away was the first thing this code did and it was wrong in a way that
took a whole flight to see: a marker viewed square on is the common case in a
small room and it is precisely the ambiguous one, so rejection meant no fixes at
all through most of a mission. The aircraft duly noticed it was not localised and
landed itself. A fix with an honest 15 cm of uncertainty along one axis is worth a
great deal more than no fix and the filter knows what to do with it.

Only a separation so large that one Gaussian cannot span it, ``max_ambiguity``,
still causes a rejection. Two modes a metre and a half apart are not an uncertain
answer, they are two different answers.

What a single marker still cannot give you is good depth: the pose rests on the
marker's apparent size, so a one-pixel error in the projected side length at
2.5 m is about 2 cm of range error. The covariance reflects that and it is why
the mission planner prefers headings that put two markers in view.

**Two or more.** Every corner of every visible marker becomes one correspondence
in a single ``SOLVEPNP_SQPNP`` problem over world-frame object points. This is
deliberately not "solve each marker and average the poses". Averaging weights a
marker at 3 m the same as one at 1 m, when the near one constrains the answer far
more strongly and it throws away the fact that two markers on perpendicular
walls fix the depth that neither fixes alone.

Covariance
----------
Not a hand-tuned constant. The pose Jacobian comes from ``cv2.projectPoints``,
the pixel noise comes from the measured reprojection residual with a floor and
the two combine as ``sigma^2 (J^T J)^-1`` in the usual way. That is then pushed
through to world-frame position and yaw by finite differences and the markers'
own survey uncertainty is added on top, because a fix cannot be more certain than
the survey it is measured against.

The result is a covariance that grows when the aircraft is far from the markers,
when it sees them obliquely and when it sees fewer of them, which is exactly the
behaviour the fusion filter and the failsafe need in order to do anything sensible
with a fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..camera import CameraModel
from ..geometry import invert_transform, make_transform, matrix_to_euler, wrap_angle
from ..types import MarkerFix, MarkerObservation
from .markers import Marker, MarkerMap

# Below this the reprojection residual is no longer a measurement of anything:
# it is the sub-pixel refinement's own noise floor. Using a smaller number would
# produce fixes the filter trusts far more than it should.
MIN_PIXEL_SIGMA = 0.35

# The residual gate, as a fraction of the marker's own projected size, with an
# absolute floor and ceiling in pixels.
#
# Relative rather than absolute, because an absolute gate is wrong at both ends
# of the range. A 3 px residual on a marker 340 px across, which is what a marker
# at 60 cm looks like, is a one per cent misfit and a perfectly good solve; the
# same 3 px on a marker 60 px across is a five per cent misfit and almost
# certainly a wrong association. A fixed 3 px gate rejected the first and
# accepted the second, which is exactly backwards.
MAX_REPROJECTION_FRACTION = 0.02
MIN_REPROJECTION_GATE = 2.0
MAX_REPROJECTION_GATE = 8.0

# Kept for callers that want a single number and used as the default when the
# projected size is not available.
MAX_REPROJECTION_RMSE = 3.0


def reprojection_gate(image_points: np.ndarray) -> float:
    """The residual a solve on these image points is allowed to have, in pixels."""
    points = np.asarray(image_points, dtype=float).reshape(-1, 4, 2)
    spans = [
        max(float(np.linalg.norm(quad[i] - quad[(i + 1) % 4])) for i in range(4))
        for quad in points
    ]
    if not spans:
        return MAX_REPROJECTION_RMSE
    return float(
        np.clip(
            MAX_REPROJECTION_FRACTION * float(np.median(spans)),
            MIN_REPROJECTION_GATE,
            MAX_REPROJECTION_GATE,
        )
    )

# Two IPPE solutions whose reprojection errors are within this factor of each
# other are not distinguishable by fit. Below it, a prior decides; without a
# prior the fix is thrown away.
AMBIGUITY_RATIO = 2.0


class PoseSolveError(RuntimeError):
    """Raised when a pose cannot be computed at all, as distinct from computed badly."""


class AmbiguousPoseError(PoseSolveError):
    """Raised when the planar two-fold ambiguity cannot be resolved."""


@dataclass(frozen=True)
class Ambiguity:
    """What the planar two-fold ambiguity did to a single-marker solve.

    Carried out of the solver so that the covariance can be widened to cover it.
    ``resolved`` means something decided between the branches: the fit, or the
    prior. When nothing decided, ``separation`` and ``direction`` describe the
    other mode the answer might equally have been.
    """

    separation: float = 0.0
    direction: Optional[np.ndarray] = None
    resolved: bool = True
    detail: str = ""
    offset: Optional[np.ndarray] = None
    """Vector from the reported branch's body position to the other one."""

    yaw_difference: float = 0.0
    """Radians between the two branches' headings."""


def _solve_single(
    marker: Marker,
    observation: MarkerObservation,
    camera: CameraModel,
    prior_position: Optional[np.ndarray],
    prior_sigma: float,
    T_bc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Ambiguity]:
    """Pose of the camera in the world from one marker.

    Returns ``(rvec, tvec, ambiguity)`` for the world-to-camera transform,
    matching the multi-marker solver's convention so the two share everything
    downstream.
    """
    object_points = marker.corners_marker().astype(np.float64)
    image_points = np.asarray(observation.corners, dtype=np.float64).reshape(4, 1, 2)
    count, rvecs, tvecs, errors = cv2.solvePnPGeneric(
        object_points,
        image_points,
        camera.matrix,
        np.asarray(camera.distortion, dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not count:
        raise PoseSolveError(f"IPPE failed for marker {marker.marker_id}")

    errors = np.asarray(errors, dtype=float).ravel()
    order = np.argsort(errors)
    candidates = []
    for index in order:
        # rvec/tvec are marker-to-camera. Compose with the marker's world pose to
        # get world-to-camera, so the covariance machinery downstream sees the
        # same parameterisation whichever solver ran.
        R_cm, _ = cv2.Rodrigues(rvecs[index])
        T_cw = make_transform(R_cm, np.asarray(tvecs[index]).ravel()) @ invert_transform(
            marker.pose
        )
        T_wb = invert_transform(T_cw) @ invert_transform(np.asarray(T_bc, dtype=float))
        yaw, _, _ = matrix_to_euler(T_wb[:3, :3])
        candidates.append((float(errors[index]), T_cw, T_wb[:3, 3].copy(), float(yaw)))

    best = candidates[0]
    ambiguity = Ambiguity()
    if len(candidates) > 1:
        second = candidates[1]
        offset = second[2] - best[2]
        separation = float(np.linalg.norm(offset))
        direction = offset / separation if separation > 1e-9 else None

        # Fit first, prior only as a tie-break. Ordering it the other way round
        # is tempting and is worse: when the fit *is* decisive, a prior with a
        # few centimetres of its own error will sometimes overrule it and pick
        # the branch that reprojects visibly badly.
        if second[0] > AMBIGUITY_RATIO * max(best[0], 1e-6):
            ambiguity = Ambiguity(
                separation=separation, direction=direction, resolved=True, detail="fit"
            )
        elif separation <= 1e-3:
            ambiguity = Ambiguity(separation=separation, resolved=True, detail="coincident")
        else:
            resolved_by_prior = False
            if prior_position is not None:
                prior = np.asarray(prior_position, dtype=float)
                distances = sorted(
                    ((float(np.linalg.norm(c[2] - prior)), c) for c in candidates),
                    key=lambda item: item[0],
                )
                # The prior only gets to decide when it actually decides: one
                # branch inside its own uncertainty and the other outside. If
                # both are plausible under the prior, picking the nearer one is
                # a coin toss dressed up as a measurement.
                gate = 3.0 * max(prior_sigma, 1e-3)
                if distances[0][0] < gate <= distances[1][0]:
                    best = distances[0][1]
                    resolved_by_prior = True
            other = candidates[1] if best is candidates[0] else candidates[0]
            ambiguity = Ambiguity(
                separation=separation,
                direction=direction,
                resolved=resolved_by_prior,
                detail="prior" if resolved_by_prior else "unresolved",
                offset=(other[2] - best[2]).copy(),
                yaw_difference=float(wrap_angle(other[3] - best[3])),
            )

    rvec_cw, _ = cv2.Rodrigues(np.ascontiguousarray(best[1][:3, :3]))
    return rvec_cw.reshape(3, 1), best[1][:3, 3].reshape(3, 1), ambiguity


def _solve_multi(
    markers: Sequence[Marker],
    observations: Sequence[MarkerObservation],
    camera: CameraModel,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pose of the camera in the world from several markers, as ``(rvec, tvec)``
    of the world-to-camera transform."""
    object_points = np.vstack([m.corners_world() for m in markers]).astype(np.float64)
    image_points = np.vstack(
        [np.asarray(o.corners, dtype=np.float64).reshape(4, 2) for o in observations]
    ).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera.matrix,
        np.asarray(camera.distortion, dtype=np.float64),
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        raise PoseSolveError("SQPnP failed")
    return rvec, tvec


def _reprojection_rmse(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera: CameraModel,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera.matrix, np.asarray(camera.distortion, dtype=np.float64)
    )
    residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def fix_covariance(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera: CameraModel,
    pixel_sigma: float,
    survey_variance: float,
    T_bc: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Position and yaw covariance of the *body* pose in the world frame.

    The chain is: pixel noise to the six-parameter camera pose, via the
    projection Jacobian; then camera pose to body position and yaw, by finite
    differences, because the analytic derivative of a rotation composition
    through a Rodrigues parameterisation is more code than it is worth for a
    quantity evaluated a few tens of times a second.

    ``survey_variance`` is added isotropically at the end. It does not go through
    the Jacobian because it is not pixel noise: it is an error in where the world
    points are and to first order it moves the answer one for one.
    """
    _, jacobian = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera.matrix,
        np.asarray(camera.distortion, dtype=np.float64),
    )
    J = np.asarray(jacobian, dtype=float)[:, :6]
    information = J.T @ J
    # A tiny ridge term. With four coplanar points seen nearly edge on the normal
    # matrix can be numerically singular and the honest response is a very large
    # covariance rather than an exception.
    ridge = 1e-9 * float(np.trace(information)) / 6.0 + 1e-12
    pose_covariance = (pixel_sigma**2) * np.linalg.inv(information + ridge * np.eye(6))

    def body_pose(parameters: np.ndarray) -> np.ndarray:
        R_cw, _ = cv2.Rodrigues(parameters[:3].reshape(3, 1))
        T_cw = make_transform(R_cw, parameters[3:])
        T_wb = invert_transform(T_cw) @ invert_transform(T_bc)
        yaw, _, _ = matrix_to_euler(T_wb[:3, :3])
        return np.concatenate([T_wb[:3, 3], [yaw]])

    parameters = np.concatenate([np.asarray(rvec).ravel(), np.asarray(tvec).ravel()])
    base = body_pose(parameters)
    G = np.zeros((4, 6))
    for i in range(6):
        step = 1e-6 * max(1.0, abs(parameters[i]))
        perturbed = parameters.copy()
        perturbed[i] += step
        G[:, i] = (body_pose(perturbed) - base) / step

    full = G @ pose_covariance @ G.T
    position_covariance = full[:3, :3] + survey_variance * np.eye(3)
    yaw_variance = float(full[3, 3]) + survey_variance / 4.0
    return position_covariance, yaw_variance


def estimate_pose(
    observations: Sequence[MarkerObservation],
    marker_map: MarkerMap,
    camera: CameraModel,
    T_bc: np.ndarray,
    timestamp: Optional[float] = None,
    max_rmse: Optional[float] = None,
    prior_position: Optional[np.ndarray] = None,
    prior_sigma: float = 0.10,
    max_ambiguity: float = 1.0,
) -> Optional[MarkerFix]:
    """Best available absolute fix from a frame's detections, or ``None``.

    ``None`` means "no usable fix" and the caller must treat that as a missing
    measurement rather than as a zero. Every reason for returning ``None`` is a
    real operational condition: nothing detected, nothing detected that is in the
    map, a solve whose residual says the association or the geometry is wrong, or
    a single-marker view whose two-fold ambiguity cannot be resolved.

    ``max_rmse`` overrides the size-relative residual gate with a fixed one in
    pixels. Leave it as ``None`` unless a test needs the gate pinned.

    ``prior_position`` is the body position the caller already believes in,
    typically the fusion filter's prediction for this timestamp. It is used only
    to choose between the two IPPE branches and never enters the fix itself, so
    it cannot drag a fix towards a wrong prior: it can only pick which of two
    discrete answers is reported and only when ``prior_sigma`` says it is
    confident enough to be entitled to.
    """
    matched: List[Tuple[Marker, MarkerObservation]] = []
    for observation in observations:
        marker = marker_map.by_payload(observation.decoded)
        if marker is not None:
            matched.append((marker, observation))
    if not matched:
        return None

    # Deterministic order, so the same frame always produces the same fix
    # regardless of the order the detector happened to report symbols in.
    matched.sort(key=lambda pair: pair[0].marker_id)
    markers = [m for m, _ in matched]
    used = [o for _, o in matched]
    stamp = float(timestamp if timestamp is not None else used[0].timestamp)

    ambiguity = Ambiguity()
    try:
        if len(markers) == 1:
            rvec, tvec, ambiguity = _solve_single(
                markers[0],
                used[0],
                camera,
                prior_position,
                prior_sigma,
                np.asarray(T_bc, dtype=float),
            )
        else:
            rvec, tvec = _solve_multi(markers, used, camera)
    except PoseSolveError:
        return None

    if not ambiguity.resolved and ambiguity.separation > max_ambiguity:
        # Too far apart to represent as one Gaussian. Two modes a metre and a
        # half apart are not an uncertain answer, they are two different answers
        # and averaging them would put the estimate somewhere neither of them is.
        return None

    object_points = np.vstack([m.corners_world() for m in markers]).astype(np.float64)
    image_points = np.vstack(
        [np.asarray(o.corners, dtype=np.float64).reshape(4, 2) for o in used]
    )
    rmse = _reprojection_rmse(object_points, image_points, rvec, tvec, camera)
    gate = reprojection_gate(image_points) if max_rmse is None else float(max_rmse)
    if not np.isfinite(rmse) or rmse > gate:
        return None

    R_cw, _ = cv2.Rodrigues(rvec)
    T_cw = make_transform(R_cw, np.asarray(tvec).ravel())
    T_wb = invert_transform(T_cw) @ invert_transform(np.asarray(T_bc, dtype=float))
    yaw, _, _ = matrix_to_euler(T_wb[:3, :3])

    survey_variance = float(np.mean([m.survey_sigma**2 for m in markers]))
    position_covariance, yaw_variance = fix_covariance(
        object_points,
        rvec,
        tvec,
        camera,
        pixel_sigma=max(rmse, MIN_PIXEL_SIGMA),
        survey_variance=survey_variance,
        T_bc=np.asarray(T_bc, dtype=float),
    )

    position = T_wb[:3, 3].copy()
    if not ambiguity.resolved and ambiguity.direction is not None:
        # Moment-match the two-mode posterior rather than picking one mode.
        #
        # This is the difference between a system that localises and one that
        # does not. A marker viewed nearly square on is the *common* case in a
        # small room and it is exactly the case where the two IPPE branches
        # reproject almost equally well: the tilt is barely observable, so half a
        # pixel of corner noise swings it by several degrees, which at 1.5 m is
        # 15 cm of lateral position. Rejecting every such fix left the aircraft
        # unlocalised for nine seconds after take-off and it landed itself,
        # correctly and uselessly.
        #
        # Reporting the better-fitting branch with a widened covariance is not
        # much better and the reason is subtle: which branch fits better is
        # decided by noise, so consecutive frames alternate between two answers
        # 30 cm apart. The filter sees a measurement that jumps back and forth,
        # its innovations blow past the chi-square gate and a third of the fixes
        # are thrown away as outliers when in truth they were all equally good.
        # That is what the second attempt did: 69 per cent acceptance and 29 cm
        # of trajectory error.
        #
        # The right answer is the one the maths gives. For two equally likely
        # modes ``s`` apart, the mean is the midpoint and the variance is the
        # within-mode variance plus ``(s / 2)^2`` along the line joining them.
        # Reporting that is stable frame to frame, honest about which direction
        # is unknown and keeps the axes the solve *does* constrain, so a
        # single-marker fix still pins down range and heading.
        direction = np.asarray(ambiguity.direction, dtype=float).reshape(3, 1)
        position = position + 0.5 * np.asarray(ambiguity.offset, dtype=float)
        position_covariance = position_covariance + (
            (ambiguity.separation / 2.0) ** 2
        ) * (direction @ direction.T)
        yaw = wrap_angle(float(yaw) + 0.5 * ambiguity.yaw_difference)
        yaw_variance = yaw_variance + (ambiguity.yaw_difference / 2.0) ** 2

    return MarkerFix(
        timestamp=stamp,
        position=position,
        yaw=float(yaw),
        marker_ids=tuple(m.marker_id for m in markers),
        reprojection_rmse=float(rmse),
        position_covariance=position_covariance,
        yaw_variance=float(yaw_variance),
        n_points=int(object_points.shape[0]),
    )


def marker_range(fix: MarkerFix, marker_map: MarkerMap) -> float:
    """Mean distance from the fix position to the markers that produced it.

    Reported in the trajectory log. Fix quality is dominated by this and by how
    many markers were used and having both in the log is what makes an accuracy
    plot against range possible after the fact rather than only in a purpose-built
    experiment.
    """
    positions = [
        marker_map.get(marker_id).position
        for marker_id in fix.marker_ids
        if marker_map.get(marker_id) is not None
    ]
    if not positions:
        return float("nan")
    return float(np.mean([np.linalg.norm(fix.position - p) for p in positions]))
