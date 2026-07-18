"""Camera-pixel to tray-plane homography and tray-to-robot rigid transform."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Homography2D:
    matrix: np.ndarray

    @classmethod
    def fit(cls, image_points: np.ndarray, plane_points: np.ndarray) -> "Homography2D":
        image_points = np.asarray(image_points, dtype=float)
        plane_points = np.asarray(plane_points, dtype=float)
        if image_points.shape != plane_points.shape or image_points.ndim != 2 or image_points.shape[1] != 2:
            raise ValueError("image_points and plane_points must both be Nx2")
        if len(image_points) < 4:
            raise ValueError("at least four point correspondences are required")
        rows: list[list[float]] = []
        targets: list[float] = []
        for (u, v), (x, y) in zip(image_points, plane_points, strict=True):
            rows.append([u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v])
            targets.append(x)
            rows.append([0.0, 0.0, 0.0, u, v, 1.0, -y * u, -y * v])
            targets.append(y)
        design = np.asarray(rows, dtype=float)
        if np.linalg.matrix_rank(design) < 8:
            raise ValueError("calibration points do not constrain a homography")
        parameters, *_ = np.linalg.lstsq(design, np.asarray(targets), rcond=None)
        matrix = np.append(parameters, 1.0).reshape(3, 3)
        return cls(matrix=matrix)

    def transform(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        single = points.ndim == 1
        if single:
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be a two-element point or Nx2 array")
        homogeneous = np.column_stack((points, np.ones(len(points))))
        mapped = (self.matrix @ homogeneous.T).T
        if np.any(np.abs(mapped[:, 2]) < 1e-12):
            raise ValueError("homography mapped a point to infinity")
        result = mapped[:, :2] / mapped[:, 2:3]
        return result[0] if single else result

    def reprojection_rmse(self, image_points: np.ndarray, plane_points: np.ndarray) -> float:
        residual = self.transform(image_points) - np.asarray(plane_points, dtype=float)
        return float(math.sqrt(np.mean(np.sum(residual**2, axis=1))))


@dataclass(frozen=True)
class RigidTransform2D:
    rotation_degrees: float
    translation: tuple[float, float]

    def transform(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        radians = math.radians(self.rotation_degrees)
        rotation = np.array(
            [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]]
        )
        return points @ rotation.T + np.asarray(self.translation)


@dataclass(frozen=True)
class CalibrationChain:
    camera_to_tray: Homography2D
    tray_to_robot: RigidTransform2D
    image_width: int
    image_height: int

    def camera_pixel_to_robot(self, point: np.ndarray) -> np.ndarray:
        tray_point = self.camera_to_tray.transform(point)
        return self.tray_to_robot.transform(tray_point)


def default_calibration(image_width: int = 640, image_height: int = 480) -> CalibrationChain:
    image_corners = np.array(
        [[0.0, 0.0], [image_width, 0.0], [image_width, image_height], [0.0, image_height]]
    )
    tray_corners = np.array([[-0.45, 0.65], [0.45, 0.65], [0.45, 0.0], [-0.45, 0.0]])
    homography = Homography2D.fit(image_corners, tray_corners)
    return CalibrationChain(
        camera_to_tray=homography,
        tray_to_robot=RigidTransform2D(rotation_degrees=-90.0, translation=(0.85, 0.20)),
        image_width=image_width,
        image_height=image_height,
    )

