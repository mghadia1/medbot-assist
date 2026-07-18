"""Adapt MedBot Vision geometry and simulator labels into a defined message."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Sequence


class PerceptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolDetectionMessage:
    name: str
    component_id: int
    centroid_x_px: float
    centroid_y_px: float
    angle_degrees: float
    confidence: float
    frame_id: str
    captured_at_seconds: float
    label_source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def adapt_synthetic_detections(
    detections: Sequence[object],
    ground_truth: dict[str, object],
    *,
    captured_at_seconds: float,
) -> list[ToolDetectionMessage]:
    """Assign simulator-truth names by globally minimizing centroid distance.

    The detector supplies geometry only. The class names are explicitly oracle labels,
    not predictions, until a real classifier is implemented.
    """

    truth_items = list(ground_truth.get("instruments", []))
    if len(detections) != len(truth_items):
        raise PerceptionError(
            f"unsafe component count: detected {len(detections)}, expected {len(truth_items)}"
        )
    best_permutation: tuple[object, ...] | None = None
    best_cost = math.inf
    for permutation in itertools.permutations(detections):
        cost = 0.0
        for detection, truth in zip(permutation, truth_items, strict=True):
            cost += math.hypot(
                float(getattr(detection, "centroid_x")) - float(truth["center_x"]),
                float(getattr(detection, "centroid_y")) - float(truth["center_y"]),
            )
        if cost < best_cost:
            best_cost = cost
            best_permutation = permutation
    if best_permutation is None:
        raise PerceptionError("no valid detection assignment")

    messages: list[ToolDetectionMessage] = []
    for detection, truth in zip(best_permutation, truth_items, strict=True):
        distance = math.hypot(
            float(getattr(detection, "centroid_x")) - float(truth["center_x"]),
            float(getattr(detection, "centroid_y")) - float(truth["center_y"]),
        )
        confidence = max(0.0, 1.0 - distance / 40.0)
        messages.append(
            ToolDetectionMessage(
                name=str(truth["name"]),
                component_id=int(getattr(detection, "component_id")),
                centroid_x_px=float(getattr(detection, "centroid_x")),
                centroid_y_px=float(getattr(detection, "centroid_y")),
                angle_degrees=float(getattr(detection, "angle_degrees")),
                confidence=confidence,
                frame_id="camera_overhead",
                captured_at_seconds=captured_at_seconds,
                label_source="simulator_ground_truth_oracle",
            )
        )
    return messages

