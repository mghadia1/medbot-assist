from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from medbot_assist.calibration import Homography2D, default_calibration
from medbot_assist.classifier import build_classifier, preprocess_crop
from medbot_assist.evaluation import evaluate_pipeline
from medbot_assist.perception import PerceptionError, ToolDetectionMessage, adapt_synthetic_detections
from medbot_assist.pipeline import AssistRequest, SafetyPolicy, run_assist
from medbot_vision.detector import detect_instruments
from medbot_vision.synthetic import generate_scene


def _messages(tmp_path: Path) -> list[ToolDetectionMessage]:
    truth = generate_scene(tmp_path, seed=7)
    detections = detect_instruments(tmp_path / "scene.png")
    return adapt_synthetic_detections(detections, truth, captured_at_seconds=0.0)


def test_homography_reprojects_calibration_points() -> None:
    image = np.array([[0.0, 0.0], [640.0, 0.0], [640.0, 480.0], [0.0, 480.0]])
    tray = np.array([[-0.45, 0.65], [0.45, 0.65], [0.45, 0.0], [-0.45, 0.0]])
    model = Homography2D.fit(image, tray)
    assert model.reprojection_rmse(image, tray) < 1e-10


def test_classifier_crop_and_forward_shape(tmp_path: Path) -> None:
    truth = generate_scene(tmp_path, seed=7)
    with Image.open(tmp_path / "scene.png") as image:
        crop = preprocess_crop(image, truth["instruments"][0]["bounding_box"])
    import torch

    logits = build_classifier()(torch.from_numpy(crop).unsqueeze(0).unsqueeze(0))
    assert crop.shape == (64, 64)
    assert logits.shape == (1, 4)


def test_adapter_marks_names_as_oracle_not_predictions(tmp_path: Path) -> None:
    messages = _messages(tmp_path)
    assert {message.name for message in messages} == {"scalpel", "forceps", "clamp", "retractor"}
    assert all(message.label_source == "simulator_ground_truth_oracle" for message in messages)


def test_adapter_rejects_component_count_mismatch(tmp_path: Path) -> None:
    truth = generate_scene(tmp_path, seed=7)
    detections = detect_instruments(tmp_path / "scene.png")
    with pytest.raises(PerceptionError, match="unsafe component count"):
        adapt_synthetic_detections(detections[:-1], truth, captured_at_seconds=0.0)


def test_requested_tool_runs_through_transform_and_planner(tmp_path: Path) -> None:
    result = run_assist(
        AssistRequest("scalpel", 0.0),
        _messages(tmp_path),
        default_calibration(),
        now_seconds=0.1,
        seed=7,
    )
    assert result.success, result.reason
    assert result.robot_target is not None
    assert result.task_result is not None
    assert result.task_result["success"]


def test_missing_stale_and_low_confidence_inputs_are_rejected(tmp_path: Path) -> None:
    messages = _messages(tmp_path)
    missing = run_assist(
        AssistRequest("needle", 0.0), messages, default_calibration(), now_seconds=0.1, seed=1
    )
    assert missing.status == "rejected"
    stale = run_assist(
        AssistRequest("scalpel", 0.0), messages, default_calibration(), now_seconds=2.0, seed=1
    )
    assert stale.status == "rejected"
    low_messages = [
        replace(message, confidence=0.2) if message.name == "scalpel" else message
        for message in messages
    ]
    low = run_assist(
        AssistRequest("scalpel", 0.0),
        low_messages,
        default_calibration(),
        now_seconds=0.1,
        seed=1,
        policy=SafetyPolicy(minimum_confidence=0.8),
    )
    assert low.status == "rejected"


def test_out_of_calibration_centroid_is_rejected(tmp_path: Path) -> None:
    messages = _messages(tmp_path)
    moved = [
        replace(message, centroid_x_px=900.0) if message.name == "scalpel" else message
        for message in messages
    ]
    result = run_assist(
        AssistRequest("scalpel", 0.0), moved, default_calibration(), now_seconds=0.1, seed=1
    )
    assert result.status == "rejected"
    assert "outside" in result.reason


def test_randomized_evaluation_writes_summary(tmp_path: Path) -> None:
    summary = evaluate_pipeline(tmp_path / "evaluation", trials=3, seed=5)
    assert summary["trials"] == 3
    assert (tmp_path / "evaluation" / "trials.csv").is_file()
    assert (tmp_path / "evaluation" / "summary.json").is_file()
