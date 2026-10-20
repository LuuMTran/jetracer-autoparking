#!/usr/bin/env python3
"""Train the planned parking policy and save a reproducible artifact bundle."""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from parking_lib.core import save_json
from parking_lib.data import validate_episode
from parking_lib.learning import (ARCHITECTURES, RESNET18_ARCH, eligible_episodes, evaluate,
                                  load_checkpoint, train)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(episodes):
    rows = []
    for episode in episodes:
        with (episode / "actions.csv").open() as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def create_plots(output, history, selected, test_report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output / "plots"
    plots.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    offset = 0
    for stage in ("frozen", "layer4"):
        values = [row for row in history if row["stage"] == stage]
        if not values:
            continue
        x = np.arange(offset + 1, offset + len(values) + 1)
        ax.plot(x, [row["train"] for row in values], marker="o", label=stage + " train")
        ax.plot(x, [row["val"] for row in values], marker="o", label=stage + " validation")
        if stage == "frozen":
            offset += len(values)
            ax.axvline(offset + .5, color="#777", linestyle="--", linewidth=1)
    ax.set(xlabel="Epoch", ylabel="Steering + throttle MSE", title="Behavior cloning loss")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "loss_curves.png", dpi=180)
    plt.close(fig)

    rows = read_rows(selected)
    steering = np.array([float(row["steering_norm"]) for row in rows])
    throttle = np.array([float(row["throttle_norm"]) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(steering, bins=31, color="#315c88")
    axes[0].set(title="Steering labels", xlabel="Normalized steering", ylabel="Frames")
    axes[1].hist(throttle, bins=31, color="#d07a43")
    axes[1].set(title="Throttle labels", xlabel="Normalized throttle", ylabel="Frames")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(plots / "action_distributions.png", dpi=180)
    plt.close(fig)

    names = [item["episode"].replace("episode_", "")[-8:] for item in test_report["episodes"]]
    steering_mse = [item["mse"][0] for item in test_report["episodes"]]
    throttle_mse = [item["mse"][1] for item in test_report["episodes"]]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - .2, steering_mse, .4, label="steering")
    ax.bar(x + .2, throttle_mse, .4, label="throttle")
    ax.set(title="Held-out episode error", xlabel="Episode suffix", ylabel="Normalized MSE",
           xticks=x, xticklabels=names)
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "test_episode_mse.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5,
                        help="Validation epochs without improvement before stopping")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default=RESNET18_ARCH)
    parser.add_argument("--split", type=Path,
                        help="Reuse an existing split.json so model comparisons use the same episodes")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    args = parser.parse_args(argv)

    all_episodes = sorted(args.dataset.glob("episode_*"))
    completed = []
    validation = []
    excluded = []
    for episode in all_episodes:
        report = validate_episode(episode)
        metadata = json.loads((episode / "metadata.json").read_text())
        if metadata.get("status") == "completed" and report["valid"]:
            completed.append(episode)
        else:
            excluded.append({"episode": episode.name, "status": metadata.get("status"),
                             "end_reason": metadata.get("end_reason"), "errors": report["errors"]})
        validation.append(report)
    if len(completed) < 3:
        raise RuntimeError("Need at least three valid completed episodes")

    manifest = {
        "dataset": str(args.dataset.resolve()),
        "dataset_was_modified": False,
        "selection_policy": "valid completed human episodes from the user-authorized dataset, excluding collision/discard",
        "episodes_found": len(all_episodes),
        "episodes_selected": len(completed),
        "episodes_excluded": excluded,
        "selected_episode_ids": [p.name for p in completed],
        "metadata_sha256_before_training": {p.name: file_sha256(p / "metadata.json") for p in all_episodes},
        "validation": validation,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "dataset_manifest.json", manifest)
    started = time.time()
    checkpoint = train(args.dataset, args.output / "model", epochs=args.epochs,
                       finetune_epochs=args.finetune_epochs, batch_size=args.batch_size,
                       sequence_length=16, stride=8, seed=args.seed, patience=args.patience,
                       device=args.device, pretrained=True,
                       include_completed_unreviewed=True, split_file=args.split,
                       architecture=args.architecture, augment=args.augment,
                       init_checkpoint=args.init_checkpoint, learning_rate=args.learning_rate,
                       encoder_learning_rate=args.encoder_learning_rate)
    test_report = evaluate(args.dataset, checkpoint, args.device)
    save_json(args.output / "test_report.json", test_report)
    history = json.loads((args.output / "model" / "history.json").read_text())
    saved = load_checkpoint(checkpoint, "cpu")
    summary = {
        "architecture": saved["architecture"],
        "checkpoint": str(checkpoint.relative_to(args.output)),
        "checkpoint_sha256": file_sha256(checkpoint),
        "device": args.device,
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "seed": args.seed,
        "sequence_length": 16,
        "stride": 8,
        "requested_epochs": args.epochs,
        "requested_finetune_epochs": args.finetune_epochs,
        "patience": args.patience,
        "completed_epochs": len(history),
        "best_validation_loss": saved["best_val_loss"],
        "training_settings": saved.get("training_settings"),
        "test_normalized_steering_mse": test_report["normalized_steering_mse"],
        "test_normalized_throttle_mse": test_report["normalized_throttle_mse"],
        "elapsed_seconds": time.time() - started,
    }
    save_json(args.output / "training_summary.json", summary)
    create_plots(args.output, history, completed, test_report)
    manifest["metadata_sha256_after_training"] = {p.name: file_sha256(p / "metadata.json") for p in all_episodes}
    manifest["dataset_was_modified"] = (manifest["metadata_sha256_before_training"] !=
                                         manifest["metadata_sha256_after_training"])
    save_json(args.output / "dataset_manifest.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
