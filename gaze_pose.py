"""Fit a head-orientation correction from same-target gaze samples."""

import math
from dataclasses import dataclass

import numpy as np

POSE_DIRECTIONS = (("turn your head slightly left", 0), ("turn your head slightly right", 0),
                   ("tilt your head slightly up", 1), ("tilt your head slightly down", 1))
MIN_TURN = 0.065
MAX_TURN = 0.43
MAX_ROLL = 0.30


def orientation_from_matrix(matrix):
    """Read yaw, pitch and roll in radians from MediaPipe's 4×4 pose transform."""
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return None
    # Strip scale and mild numerical skew so only the rotation influences gaze.
    u, scales, v = np.linalg.svd(transform[:3, :3])
    rotation = u @ v
    if scales[-1] < 1e-4 or scales[0] / scales[-1] > 1.35 or np.linalg.det(rotation) <= 0:
        return None
    forward = rotation[:, 2]
    if forward[2] < 0.5:
        return None
    return (math.atan2(forward[0], forward[2]),
            math.atan2(forward[1], math.hypot(forward[0], forward[2])),
            math.atan2(rotation[1, 0], rotation[0, 0]))


@dataclass(frozen=True)
class HeadCompensation:
    reference: np.ndarray
    coefficients: np.ndarray
    limits: np.ndarray

    def within_range(self, orientation):
        offset = np.asarray(orientation) - self.reference
        lower, upper = self.limits
        margin = np.maximum(0.12, (upper - lower) * 0.28)
        return bool(np.all(offset[:2] >= np.maximum(lower - margin, -MAX_TURN))
                    and np.all(offset[:2] <= np.minimum(upper + margin, MAX_TURN))
                    and abs(offset[2]) <= MAX_ROLL)

    def correct(self, iris, orientation):
        offset = np.asarray(orientation)[..., :2] - self.reference[:2]
        # Do not extrapolate unlimited gains past the sampled head orientations.
        return np.asarray(iris) - np.clip(offset, *self.limits) @ self.coefficients


def fit_head_compensation(reference, iris, samples, gaze_map):
    """Learn iris drift at one fixed screen point, retaining only a validated correction."""
    if len(samples) != len(POSE_DIRECTIONS):
        return None
    offsets = np.array([orientation[:2] for orientation, _ in samples]) - reference[:2]
    readings = np.array([gaze for _, gaze in samples])
    # The same screen point must be sampled on both sides of each axis.
    if np.any(offsets.max(axis=0) < MIN_TURN) or np.any(offsets.min(axis=0) > -MIN_TURN):
        return None
    if np.linalg.svd(offsets, compute_uv=False)[-1] < MIN_TURN:
        return None

    def fit(delta, drift):
        # Shrink unverified gains rather than fitting all four points exactly.
        return np.linalg.solve(delta.T @ delta + 0.001 * np.eye(2), delta.T @ drift)

    center_target = gaze_map.predict(iris)
    baseline_errors = np.linalg.norm(gaze_map.predict(readings) - center_target, axis=1)
    if np.mean(baseline_errors) < 0.018:
        return None
    corrected_errors = []
    for i in range(len(offsets)):
        other = np.arange(len(offsets)) != i
        coefficients = fit(offsets[other], readings[other] - iris)
        predicted = gaze_map.predict(readings[i] - offsets[i] @ coefficients)
        corrected_errors.append(np.linalg.norm(predicted - center_target))
    corrected_errors = np.array(corrected_errors)
    if (np.mean(corrected_errors) >= 0.75 * np.mean(baseline_errors)
            or np.any(corrected_errors > baseline_errors + 0.045)):
        return None
    limits = np.array((np.min(offsets, axis=0), np.max(offsets, axis=0)))
    coefficients = fit(offsets, readings - iris)
    if np.any(np.linalg.norm(offsets @ coefficients, axis=1) > 0.18):
        return None
    return HeadCompensation(np.asarray(reference), coefficients, limits)
