"""One-command cross-platform demo and repeatable integration evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import default_calibration
from .classifier import (
    classify_detections,
    generate_classification_dataset,
    train_classifier,
)
from .evaluation import evaluate_pipeline
from .perception import adapt_synthetic_detections
from .pipeline import AssistRequest, run_assist


def _demo(
    output: Path,
    requested_tool: str,
    seed: int,
    classifier_checkpoint: Path | None,
) -> dict[str, object]:
    try:
        from medbot_vision.detector import detect_instruments, save_annotation, save_detections
        from medbot_vision.synthetic import generate_scene
        from surgiarm.task import PickPlaceScenario
        from surgiarm.visualize import render_task
    except ImportError as error:
        raise RuntimeError(
            "Install the local ../medbot-vision and ../surgiarm-sim projects first."
        ) from error

    output.mkdir(parents=True, exist_ok=True)
    vision_dir = output / "vision"
    truth = generate_scene(vision_dir, seed=seed)
    detections = detect_instruments(vision_dir / "scene.png")
    save_detections(detections, vision_dir / "detections.json")
    save_annotation(vision_dir / "scene.png", detections, vision_dir / "annotated.png")
    messages = (
        classify_detections(
            vision_dir / "scene.png",
            detections,
            classifier_checkpoint,
            captured_at_seconds=0.0,
        )
        if classifier_checkpoint is not None
        else adapt_synthetic_detections(detections, truth, captured_at_seconds=0.0)
    )
    result = run_assist(
        AssistRequest(requested_tool, requested_at_seconds=0.0),
        messages,
        default_calibration(),
        now_seconds=0.1,
        seed=seed,
    )
    (output / "detection-messages.json").write_text(
        json.dumps([message.to_dict() for message in messages], indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "assist-result.json").write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    if result.task_result is not None and result.robot_target is not None:
        # Re-run the deterministic scenario to retain typed phase snapshots for rendering.
        from surgiarm.task import run_pick_place

        typed_task = run_pick_place(
            PickPlaceScenario(object_position=tuple(result.robot_target)), seed=seed
        )
        render_task(
            typed_task,
            PickPlaceScenario(object_position=tuple(result.robot_target)),
            output / "robot-frames",
        )
    return result.to_dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MedBot Assist tray-handling integration")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run vision-to-plan integration for one requested tool")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--tool", default="scalpel")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--classifier", type=Path)
    data = commands.add_parser("make-classifier-data", help="generate synthetic train/test tool crops")
    data.add_argument("--output", type=Path, default=Path("outputs/classifier-data"))
    data.add_argument("--train-per-class", type=int, default=48)
    data.add_argument("--test-per-class", type=int, default=12)
    data.add_argument("--seed", type=int, default=211)
    train = commands.add_parser("train-classifier", help="train and evaluate the four-class CNN")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("outputs/classifier"))
    train.add_argument("--epochs", type=int, default=24)
    train.add_argument("--batch-size", type=int, default=24)
    train.add_argument("--seed", type=int, default=313)
    evaluation = commands.add_parser("evaluate", help="run randomized end-to-end trials")
    evaluation.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    evaluation.add_argument("--trials", type=int, default=20)
    evaluation.add_argument("--seed", type=int, default=101)
    evaluation.add_argument("--classifier", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        print(json.dumps(_demo(args.output, args.tool, args.seed, args.classifier), indent=2))
    elif args.command == "make-classifier-data":
        path = generate_classification_dataset(
            args.output,
            train_per_class=args.train_per_class,
            test_per_class=args.test_per_class,
            seed=args.seed,
        )
        print(path)
    elif args.command == "train-classifier":
        print(
            json.dumps(
                train_classifier(
                    args.data,
                    args.output,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=args.seed,
                ),
                indent=2,
            )
        )
    elif args.command == "evaluate":
        print(
            json.dumps(
                evaluate_pipeline(
                    args.output,
                    trials=args.trials,
                    seed=args.seed,
                    classifier_checkpoint=args.classifier,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
