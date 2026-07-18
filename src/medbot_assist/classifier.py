"""Synthetic-data CNN for four tray-tool classes."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .perception import ToolDetectionMessage

TOOL_CLASSES = ("scalpel", "forceps", "clamp", "retractor")


def preprocess_crop(
    image: Image.Image,
    bounding_box: Sequence[int],
    *,
    output_size: int = 64,
    padding_fraction: float = 0.18,
) -> np.ndarray:
    if len(bounding_box) != 4:
        raise ValueError("bounding_box must contain left, top, right, bottom")
    left, top, right, bottom = (int(value) for value in bounding_box)
    if right < left or bottom < top:
        raise ValueError("bounding_box has negative width or height")
    width = right - left + 1
    height = bottom - top + 1
    padding = max(3, round(max(width, height) * padding_fraction))
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width - 1, right + padding)
    bottom = min(image.height - 1, bottom + padding)
    crop = image.convert("L").crop((left, top, right + 1, bottom + 1))
    canvas_size = max(crop.width, crop.height)
    canvas = Image.new("L", (canvas_size, canvas_size), color=35)
    canvas.paste(crop, ((canvas_size - crop.width) // 2, (canvas_size - crop.height) // 2))
    resized = canvas.resize((output_size, output_size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def generate_classification_dataset(
    output_dir: Path,
    *,
    train_per_class: int = 48,
    test_per_class: int = 12,
    seed: int = 211,
    image_size: int = 128,
    crop_size: int = 64,
) -> Path:
    try:
        from medbot_vision.synthetic import DEFAULT_INSTRUMENTS, InstrumentSpec, generate_scene
    except ImportError as error:
        raise RuntimeError("Install ../medbot-vision before generating classifier data") from error
    if train_per_class < 4 or test_per_class < 1:
        raise ValueError("use at least four training and one test sample per class")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    class_to_spec = {spec.name: spec for spec in DEFAULT_INSTRUMENTS}
    arrays: dict[str, list[np.ndarray | int | str]] = {
        "train_images": [],
        "train_labels": [],
        "test_images": [],
        "test_labels": [],
    }
    records: list[dict[str, object]] = []
    for split, count in (("train", train_per_class), ("test", test_per_class)):
        for label, name in enumerate(TOOL_CLASSES):
            base = class_to_spec[name]
            for index in range(count):
                scene_dir = output_dir / "generated-scenes" / split / name / f"sample-{index:03d}"
                scale = float(rng.uniform(0.82, 1.14))
                spec = InstrumentSpec(
                    name=name,
                    center_x=image_size / 2 + float(rng.uniform(-8.0, 8.0)),
                    center_y=image_size / 2 + float(rng.uniform(-8.0, 8.0)),
                    length_px=min(base.length_px * scale * 0.66, image_size * 0.78),
                    width_px=min(base.width_px * float(rng.uniform(0.85, 1.15)), image_size * 0.35),
                    angle_degrees=float(rng.uniform(-80.0, 80.0)),
                )
                truth = generate_scene(
                    scene_dir,
                    seed=seed * 1_000_000 + label * 10_000 + index + (0 if split == "train" else 500_000),
                    image_size=(image_size, image_size),
                    instruments=(spec,),
                    noise_sigma=float(rng.uniform(5.0, 13.0)),
                    background_value=int(rng.integers(25, 48)),
                    instrument_value=int(rng.integers(185, 236)),
                )
                with Image.open(scene_dir / "scene.png") as source:
                    crop = preprocess_crop(
                        source,
                        truth["instruments"][0]["bounding_box"],
                        output_size=crop_size,
                    )
                arrays[f"{split}_images"].append(crop)
                arrays[f"{split}_labels"].append(label)
                records.append(
                    {
                        "split": split,
                        "class": name,
                        "sample": index,
                        "scene": str(scene_dir.relative_to(output_dir)),
                    }
                )
    dataset_path = output_dir / "tool-crops.npz"
    np.savez_compressed(
        dataset_path,
        train_images=np.asarray(arrays["train_images"], dtype=np.float32),
        train_labels=np.asarray(arrays["train_labels"], dtype=np.int64),
        test_images=np.asarray(arrays["test_images"], dtype=np.float32),
        test_labels=np.asarray(arrays["test_labels"], dtype=np.int64),
        classes=np.asarray(TOOL_CLASSES),
    )
    metadata = {
        "kind": "synthetic tool-crop classification dataset",
        "medical_data": False,
        "seed": seed,
        "train_per_class": train_per_class,
        "test_per_class": test_per_class,
        "classes": list(TOOL_CLASSES),
        "crop_size": crop_size,
        "records": records,
    }
    (output_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return dataset_path


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("Install the 'ml' extra to use the tool classifier") from error
    return torch, nn


def build_classifier(class_count: int = len(TOOL_CLASSES)):
    torch, nn = _torch_modules()

    class ToolClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 12, 5, padding=2),
                nn.BatchNorm2d(12),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(12, 24, 3, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Linear(32 * 4 * 4, class_count)

        def forward(self, inputs):
            features = self.features(inputs)
            return self.classifier(features.flatten(start_dim=1))

    return ToolClassifier()


def _accuracy(logits, labels) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def train_classifier(
    dataset_path: Path,
    output_dir: Path,
    *,
    epochs: int = 24,
    batch_size: int = 24,
    learning_rate: float = 1e-3,
    seed: int = 313,
) -> dict[str, object]:
    torch, _nn = _torch_modules()
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = np.load(dataset_path, allow_pickle=False)
    train_images = torch.from_numpy(data["train_images"]).unsqueeze(1)
    train_labels = torch.from_numpy(data["train_labels"])
    test_images = torch.from_numpy(data["test_images"]).unsqueeze(1)
    test_labels = torch.from_numpy(data["test_labels"])
    classes = [str(value) for value in data["classes"].tolist()]
    permutation = torch.randperm(len(train_images), generator=torch.Generator().manual_seed(seed))
    validation_count = max(len(classes), round(len(train_images) * 0.20))
    validation_indices = permutation[:validation_count]
    fit_indices = permutation[validation_count:]
    model = build_classifier(len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_validation_loss = float("inf")
    history: list[dict[str, float | int]] = []
    checkpoint_path = output_dir / "tool-classifier.pt"
    generator = torch.Generator().manual_seed(seed + 1)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_order = fit_indices[torch.randperm(len(fit_indices), generator=generator)]
        total_loss = 0.0
        for start in range(0, len(epoch_order), batch_size):
            indices = epoch_order[start : start + batch_size]
            logits = model(train_images[indices])
            loss = criterion(logits, train_labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            validation_logits = model(train_images[validation_indices])
            validation_accuracy = _accuracy(validation_logits, train_labels[validation_indices])
            validation_loss = float(
                criterion(validation_logits, train_labels[validation_indices]).item()
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(fit_indices),
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_accuracy > best_accuracy or (
            validation_accuracy == best_accuracy and validation_loss < best_validation_loss
        ):
            best_accuracy = validation_accuracy
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "crop_size": int(train_images.shape[-1]),
                    "selected_epoch": epoch,
                    "validation_accuracy": validation_accuracy,
                    "validation_loss": validation_loss,
                },
                checkpoint_path,
            )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        test_logits = model(test_images)
        test_probabilities = torch.softmax(test_logits, dim=1)
        predictions = test_logits.argmax(dim=1)
    confusion = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
    for truth, prediction in zip(test_labels, predictions, strict=True):
        confusion[int(truth), int(prediction)] += 1
    test_accuracy = _accuracy(test_logits, test_labels)
    confidence = test_probabilities.max(dim=1).values
    results: dict[str, object] = {
        "classes": classes,
        "fit_samples": len(fit_indices),
        "validation_samples": len(validation_indices),
        "test_samples": len(test_images),
        "selected_epoch": int(checkpoint["selected_epoch"]),
        "validation_accuracy": float(checkpoint["validation_accuracy"]),
        "validation_loss": float(checkpoint["validation_loss"]),
        "test_accuracy": test_accuracy,
        "mean_test_confidence": float(confidence.mean()),
        "confusion_matrix_rows_truth_columns_prediction": confusion.tolist(),
        "warning": "Synthetic classification accuracy is not real-image or medical evidence.",
    }
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    return results


@dataclass(frozen=True)
class LoadedClassifier:
    model: object
    classes: tuple[str, ...]
    crop_size: int


def load_classifier(checkpoint_path: Path) -> LoadedClassifier:
    torch, _nn = _torch_modules()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classes = tuple(str(value) for value in checkpoint["classes"])
    model = build_classifier(len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return LoadedClassifier(model=model, classes=classes, crop_size=int(checkpoint["crop_size"]))


def classify_detections(
    image_path: Path,
    detections: Sequence[object],
    checkpoint_path: Path,
    *,
    captured_at_seconds: float,
) -> list[ToolDetectionMessage]:
    torch, _nn = _torch_modules()
    loaded = load_classifier(checkpoint_path)
    with Image.open(image_path) as image:
        crops = [
            preprocess_crop(image, getattr(detection, "bounding_box"), output_size=loaded.crop_size)
            for detection in detections
        ]
    inputs = torch.from_numpy(np.asarray(crops, dtype=np.float32)).unsqueeze(1)
    with torch.no_grad():
        probabilities = torch.softmax(loaded.model(inputs), dim=1)
    messages: list[ToolDetectionMessage] = []
    for detection, distribution in zip(detections, probabilities, strict=True):
        confidence, class_index = distribution.max(dim=0)
        messages.append(
            ToolDetectionMessage(
                name=loaded.classes[int(class_index)],
                component_id=int(getattr(detection, "component_id")),
                centroid_x_px=float(getattr(detection, "centroid_x")),
                centroid_y_px=float(getattr(detection, "centroid_y")),
                angle_degrees=float(getattr(detection, "angle_degrees")),
                confidence=float(confidence),
                frame_id="camera_overhead",
                captured_at_seconds=captured_at_seconds,
                label_source="synthetic_cnn_prediction",
            )
        )
    return messages
