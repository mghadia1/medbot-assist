"""Randomized end-to-end trials across generation, detection, transform, and planning."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .calibration import default_calibration
from .classifier import classify_detections
from .perception import adapt_synthetic_detections
from .pipeline import AssistRequest, run_assist


def evaluate_pipeline(
    output_dir: Path,
    *,
    trials: int = 20,
    seed: int = 101,
    classifier_checkpoint: Path | None = None,
) -> dict[str, object]:
    try:
        from medbot_vision.detector import detect_instruments, save_detections
        from medbot_vision.synthetic import DEFAULT_INSTRUMENTS, InstrumentSpec, generate_scene
    except ImportError as error:
        raise RuntimeError(
            "MedBot Vision is required. Install ../medbot-vision into this environment."
        ) from error

    if trials < 1:
        raise ValueError("trials must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    calibration = default_calibration()
    rows: list[dict[str, object]] = []
    tool_names = [spec.name for spec in DEFAULT_INSTRUMENTS]
    for index in range(trials):
        trial_dir = output_dir / "trials" / f"trial-{index:03d}"
        varied = tuple(
            InstrumentSpec(
                name=spec.name,
                center_x=spec.center_x + float(rng.uniform(-12.0, 12.0)),
                center_y=spec.center_y + float(rng.uniform(-12.0, 12.0)),
                length_px=spec.length_px,
                width_px=spec.width_px,
                angle_degrees=spec.angle_degrees + float(rng.uniform(-8.0, 8.0)),
            )
            for spec in DEFAULT_INSTRUMENTS
        )
        truth = generate_scene(trial_dir, seed=seed * 10_000 + index, instruments=varied)
        detections = detect_instruments(trial_dir / "scene.png")
        save_detections(detections, trial_dir / "detections.json")
        messages = (
            classify_detections(
                trial_dir / "scene.png",
                detections,
                classifier_checkpoint,
                captured_at_seconds=0.0,
            )
            if classifier_checkpoint is not None
            else adapt_synthetic_detections(detections, truth, captured_at_seconds=0.0)
        )
        oracle_messages = adapt_synthetic_detections(
            detections, truth, captured_at_seconds=0.0
        )
        oracle_names = {
            message.component_id: message.name for message in oracle_messages
        }
        component_class_correct = sum(
            message.name == oracle_names[message.component_id] for message in messages
        )
        requested = tool_names[index % len(tool_names)]
        result = run_assist(
            AssistRequest(requested, requested_at_seconds=0.0),
            messages,
            calibration,
            now_seconds=0.1,
            seed=seed * 10_000 + index,
        )
        requested_predictions = [message for message in messages if message.name == requested]
        truth_item = next(item for item in truth["instruments"] if item["name"] == requested)
        selected = requested_predictions[0] if len(requested_predictions) == 1 else None
        centroid_error = (
            float(
                np.linalg.norm(
                    np.array([selected.centroid_x_px, selected.centroid_y_px])
                    - np.array([truth_item["center_x"], truth_item["center_y"]])
                )
            )
            if selected is not None
            else None
        )
        true_robot_target = calibration.camera_pixel_to_robot(
            np.array([truth_item["center_x"], truth_item["center_y"]])
        )
        target_error = (
            float(np.linalg.norm(np.asarray(result.robot_target) - true_robot_target))
            if result.robot_target is not None
            else None
        )
        rows.append(
            {
                "trial": index,
                "requested_tool": requested,
                "status": result.status,
                "success": result.success,
                "detected_components": len(detections),
                "component_class_correct": component_class_correct,
                "confidence": selected.confidence if selected is not None else "",
                "centroid_error_px": centroid_error if centroid_error is not None else "",
                "robot_target_error": target_error if target_error is not None else "",
                "reason": result.reason,
            }
        )
        (trial_dir / "assist-result.json").write_text(
            json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    successes = [row for row in rows if row["success"]]
    centroid_errors = [row["centroid_error_px"] for row in rows if row["centroid_error_px"] != ""]
    total_components = int(sum(int(row["detected_components"]) for row in rows))
    correct_components = int(sum(int(row["component_class_correct"]) for row in rows))
    summary: dict[str, object] = {
        "trials": trials,
        "seed": seed,
        "successes": len(successes),
        "failures": trials - len(successes),
        "success_rate": len(successes) / trials,
        "component_classification_accuracy": correct_components / total_components,
        "correct_component_classes": correct_components,
        "total_components": total_components,
        "mean_centroid_error_px": float(np.mean(centroid_errors)) if centroid_errors else None,
        "mean_robot_target_error": float(
            np.mean([row["robot_target_error"] for row in successes])
        ) if successes else None,
        "label_source": (
            "synthetic CNN prediction"
            if classifier_checkpoint is not None
            else "simulator ground-truth oracle; tool classes are not predicted"
        ),
        "limitation": (
            "synthetic learned classes, classical geometry, and 2D kinematics; no real images or ROS/Gazebo"
            if classifier_checkpoint is not None
            else "synthetic oracle classes, classical geometry, and 2D kinematics; no learned class model or ROS/Gazebo"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
