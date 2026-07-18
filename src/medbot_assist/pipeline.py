"""Safety-gated request selection, transforms, and planar manipulation handoff."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .calibration import CalibrationChain
from .perception import ToolDetectionMessage


@dataclass(frozen=True)
class AssistRequest:
    tool_name: str
    requested_at_seconds: float


@dataclass(frozen=True)
class SafetyPolicy:
    minimum_confidence: float = 0.80
    maximum_detection_age_seconds: float = 1.0


@dataclass
class AssistResult:
    status: str
    reason: str
    requested_tool: str
    selected_detection: dict[str, object] | None
    robot_target: list[float] | None
    task_result: dict[str, object] | None

    @property
    def success(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _reject(request: AssistRequest, reason: str) -> AssistResult:
    return AssistResult(
        status="rejected",
        reason=reason,
        requested_tool=request.tool_name,
        selected_detection=None,
        robot_target=None,
        task_result=None,
    )


def run_assist(
    request: AssistRequest,
    detections: list[ToolDetectionMessage],
    calibration: CalibrationChain,
    *,
    now_seconds: float,
    seed: int,
    policy: SafetyPolicy | None = None,
) -> AssistResult:
    policy = policy or SafetyPolicy()
    matches = [detection for detection in detections if detection.name == request.tool_name]
    if not matches:
        return _reject(request, f"requested tool {request.tool_name!r} was not detected")
    if len(matches) > 1:
        return _reject(request, f"requested tool {request.tool_name!r} has ambiguous duplicate detections")
    selected = matches[0]
    age = now_seconds - selected.captured_at_seconds
    if age < 0:
        return _reject(request, "detection timestamp is in the future")
    if age > policy.maximum_detection_age_seconds:
        return _reject(request, f"detection is stale: age={age:.3f}s")
    if selected.confidence < policy.minimum_confidence:
        return _reject(
            request,
            f"detection confidence {selected.confidence:.3f} is below {policy.minimum_confidence:.3f}",
        )

    pixel = np.array([selected.centroid_x_px, selected.centroid_y_px])
    if not (0 <= pixel[0] <= calibration.image_width and 0 <= pixel[1] <= calibration.image_height):
        return _reject(request, f"centroid {pixel.tolist()} lies outside the calibrated image")
    robot_target = calibration.camera_pixel_to_robot(pixel)

    try:
        from surgiarm.task import PickPlaceScenario, run_pick_place
    except ImportError as error:
        raise RuntimeError(
            "SurgiArm Sim is required. Install ../surgiarm-sim into this environment."
        ) from error

    scenario = PickPlaceScenario(object_position=tuple(robot_target))
    task = run_pick_place(scenario, seed=seed)
    if not task.success:
        return AssistResult(
            status="planning_failed",
            reason=task.failure_reason or "unknown manipulation failure",
            requested_tool=request.tool_name,
            selected_detection=selected.to_dict(),
            robot_target=robot_target.tolist(),
            task_result=task.to_dict(),
        )
    return AssistResult(
        status="success",
        reason="requested tool selected, transformed, and placed in the planar simulation",
        requested_tool=request.tool_name,
        selected_detection=selected.to_dict(),
        robot_target=robot_target.tolist(),
        task_result=task.to_dict(),
    )

