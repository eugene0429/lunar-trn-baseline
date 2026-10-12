"""Pinhole camera model and pose helpers.

Camera frame: +X right, +Y down, +Z along the optical axis (OpenCV convention).
``T_world_cam`` maps camera coordinates to world coordinates (camera-to-world).

Pixel convention: integer pixel index ``(col, row)`` has continuous image coordinate
``u = col + pixel_center``, ``v = row + pixel_center`` in the frame in which ``K`` is
defined. OpenCV uses ``pixel_center = 0``. A renderer that puts the principal point of a
symmetric frustum at ``W / 2`` (as StereoLunar's ``cx = 256`` for 512 px does) implies
``pixel_center = 0.5``; the value used for StereoLunar is fixed by the V1 test
(``docs/decisions.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PinholeCamera:
    K: np.ndarray
    width: int
    height: int
    pixel_center: float = 0.5

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    def pixel_grid(self, step: int = 1, offset: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Column and row index grids (each (H', W')) sampled every ``step`` pixels."""
        start = (step - 1) / 2.0 if offset is None else offset
        cols = np.arange(start, self.width, step, dtype=np.float64)
        rows = np.arange(start, self.height, step, dtype=np.float64)
        return np.meshgrid(cols, rows)

    def rays_cam(self, cols, rows) -> np.ndarray:
        """Unit ray directions (..., 3) in the camera frame through pixel indices (cols, rows)."""
        u = np.asarray(cols, dtype=np.float64) + self.pixel_center
        v = np.asarray(rows, dtype=np.float64) + self.pixel_center
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        d = np.stack([x, y, np.ones_like(x)], axis=-1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def rays_world(self, T_world_cam: np.ndarray, cols, rows) -> tuple[np.ndarray, np.ndarray]:
        """Ray origin (3,) and unit directions (..., 3) in the world frame."""
        T = np.asarray(T_world_cam, dtype=np.float64)
        d = self.rays_cam(cols, rows) @ T[:3, :3].T
        return T[:3, 3].copy(), d

    def project(self, T_world_cam: np.ndarray, points_world) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project world points (..., 3) to fractional pixel indices (cols, rows) and camera depth z."""
        T = np.asarray(T_world_cam, dtype=np.float64)
        R = T[:3, :3]
        p_cam = (np.asarray(points_world, dtype=np.float64) - T[:3, 3]) @ R
        z = p_cam[..., 2]
        u = self.fx * p_cam[..., 0] / z + self.cx
        v = self.fy * p_cam[..., 1] / z + self.cy
        return u - self.pixel_center, v - self.pixel_center, z

    def gsd(self, distance_m: float) -> float:
        """Ground sample distance (m/px) at boresight range ``distance_m`` for a flat target."""
        return distance_m / self.fx


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix (SVD), used to clean float32 rotations from datasets."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angle (deg) of the relative rotation R_a^T R_b."""
    R = np.asarray(R_a).T @ np.asarray(R_b)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(c)))
