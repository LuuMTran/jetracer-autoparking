#!/usr/bin/env python3
"""Label terminal parking frames and train a frozen-policy recurrent stop head."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from parking_lib.core import save_json
from parking_lib.learning import (ParkingPolicy, StopHead, file_sha256, image_tensor,
                                  load_checkpoint)


def terminal_labels(rows, max_neutral):
    labels = [0] * len(rows)
    trailing = 0
    for row in reversed(rows):
        if abs(float(row["throttle_norm"])) <= .05:
            trailing += 1
        else:
            break
    positive_rows = min(trailing, max_neutral)
    if positive_rows:
        labels[-positive_rows:] = [1] * positive_rows
    return labels, trailing, positive_rows


def load_episode_features(model, episode, device, chunk_size, max_neutral, final_repeats):
    meta = json.loads((episode / "metadata.json").read_text())
    if meta.get("status") != "completed" or meta.get("end_reason") != "operator_finished":
        raise ValueError("Stop labels require completed operator-finished episodes: " + episode.name)
    with (episode / "actions.csv").open() as source:
        rows = list(csv.DictReader(source))
    if not rows or not any(abs(float(row["throttle_norm"])) > .05 for row in rows):
        raise ValueError("Stop labels require an episode that moved: " + episode.name)
    labels, trailing, positive_rows = terminal_labels(rows, max_neutral)
    items = [(episode / row["frame"],
              [float(row["prev_steering_norm"]), float(row["prev_throttle_norm"])])
             for row in rows]
    final_path = episode / "final.jpg"
    if not final_path.exists():
        raise ValueError("Missing final stopped image: " + str(final_path))
    items.extend((final_path, [0., 0.]) for _ in range(final_repeats))
    labels.extend([1] * final_repeats)

    recurrent_parts = []
    hidden = None
    with torch.no_grad():
        for start in range(0, len(items), chunk_size):
            part = items[start:start + chunk_size]
            images = []
            for path, _ in part:
                with Image.open(str(path)) as image:
                    images.append(image_tensor(image.convert("RGB")))
            previous = torch.tensor([previous for _, previous in part], dtype=torch.float32)
            image_batch = torch.stack(images)[None].to(device)
            previous_batch = previous[None].to(device)
            recurrent, hidden = model.recurrent_features(image_batch, previous_batch, hidden)
            recurrent_parts.append(recurrent[0].cpu())
            hidden = hidden.detach()
    return (torch.cat(recurrent_parts), torch.tensor(labels, dtype=torch.float32),
            {"episode": episode.name, "action_rows": len(rows), "trailing_neutral_rows": trailing,
             "labeled_terminal_rows": positive_rows, "final_image_repeats": final_repeats,
             "positive_labels": int(sum(labels)), "total_labels": len(labels)})


def extract_split(model, root, names, device, chunk_size, max_neutral, final_repeats):
    features, labels, episodes = [], [], []
    offset = 0
    for index, name in enumerate(names):
        feature, label, audit = load_episode_features(
            model, root / name, device, chunk_size, max_neutral, final_repeats)
        features.append(feature)
        labels.append(label)
        audit["start"] = offset
        offset += len(label)
        audit["end"] = offset
        episodes.append(audit)
        print("features {}/{} {}".format(index + 1, len(names), name), flush=True)
    return torch.cat(features), torch.cat(labels), episodes


def frame_metrics(probabilities, labels, threshold):
    predicted = probabilities >= threshold
    truth = labels >= .5
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    tn = int((~predicted & ~truth).sum())
    precision = tp / float(tp + fp) if tp + fp else 0.
    recall = tp / float(tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / float(tp + fp + fn + tn)}


def first_trigger(values, threshold, consecutive, minimum_index):
    streak = 0
    for index, value in enumerate(values):
        if index < minimum_index:
            streak = 0
            continue
        streak = streak + 1 if value >= threshold else 0
        if streak >= consecutive:
            return index
    return None


def episode_metrics(probabilities, labels, episodes, threshold, consecutive, minimum_frames):
    detected = early = missed = 0
    details = []
    for episode in episodes:
        start, end = episode["start"], episode["end"]
        values = probabilities[start:end]
        truth = labels[start:end]
        positive_indices = np.flatnonzero(truth >= .5)
        onset = int(positive_indices[0])
        trigger = first_trigger(values, threshold, consecutive, minimum_frames - 1)
        if trigger is None:
            outcome = "missed"
            missed += 1
        elif trigger < onset:
            outcome = "early"
            early += 1
        else:
            outcome = "detected"
            detected += 1
        details.append({"episode": episode["episode"], "terminal_onset_index": onset,
                        "trigger_index": trigger, "outcome": outcome,
                        "terminal_max_probability": float(values[onset:].max()),
                        "preterminal_max_probability": float(values[:onset].max()) if onset else None})
    count = len(episodes)
    return {"episodes": count, "detected": detected, "early": early, "missed": missed,
            "detection_rate": detected / float(count), "early_rate": early / float(count),
            "details": details}


def probabilities(head, features, device, batch_size):
    output = []
    head.eval()
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(features), batch_size=batch_size):
            output.append(torch.sigmoid(head(batch.to(device))).cpu())
    return torch.cat(output).numpy()


def evaluate_thresholds(head, features, labels, episodes, device, batch_size,
                        consecutive, minimum_frames):
    probs = probabilities(head, features, device, batch_size)
    truth = labels.numpy()
    candidates = []
    for value in np.arange(.50, .951, .05):
        threshold = round(float(value), 2)
        frames = frame_metrics(probs, truth, float(threshold))
        sequence = episode_metrics(probs, truth, episodes, float(threshold),
                                   consecutive, minimum_frames)
        candidates.append({"threshold": float(threshold), "frame": frames, "episode": sequence})
    zero_early = [item for item in candidates if item["episode"]["early"] == 0]
    pool = zero_early if zero_early else candidates
    best_detected = max(item["episode"]["detected"] for item in pool)
    near_best = [item for item in pool if item["episode"]["detected"] >= best_detected - 1]
    selected = max(near_best, key=lambda item: (item["frame"]["precision"],
                                                item["frame"]["f1"], item["threshold"]))
    return probs, candidates, selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-terminal-neutral", type=int, default=5)
    parser.add_argument("--final-repeats", type=int, default=4)
    parser.add_argument("--consecutive-frames", type=int, default=4)
    parser.add_argument("--minimum-elapsed-seconds", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite: " + str(args.output))
    if min(args.epochs, args.patience, args.batch_size, args.chunk_size,
           args.max_terminal_neutral, args.final_repeats, args.consecutive_frames) < 1:
        raise ValueError("Counts must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True)
    started = time.time()

    saved = load_checkpoint(args.checkpoint, args.device)
    architecture = saved["architecture"]
    model = ParkingPolicy(pretrained=False, architecture=architecture).to(args.device).eval()
    model.load_state_dict(saved["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad = False
    original_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    split = saved["split"]
    extracted = {}
    label_audit = {"rule": {
        "positive": "up to the final N zero-throttle action rows plus repeated final stopped image",
        "negative": "all earlier action rows, including initial and intermediate neutral rows",
        "required_episode": "completed, operator_finished, and contains a nonzero throttle command",
        "neutral_threshold": .05, "max_terminal_neutral_rows": args.max_terminal_neutral,
        "final_image_repeats": args.final_repeats}, "splits": {}}
    for phase in ("train", "val", "test"):
        names = []
        excluded = []
        for name in split[phase]:
            episode = args.dataset / name
            try:
                metadata = json.loads((episode / "metadata.json").read_text())
                with (episode / "actions.csv").open() as source:
                    rows = list(csv.DictReader(source))
                if metadata.get("status") != "completed" or metadata.get("end_reason") != "operator_finished":
                    raise ValueError("episode was not completed by the operator")
                if not rows or not any(abs(float(row["throttle_norm"])) > .05 for row in rows):
                    raise ValueError("episode contains no movement")
                if not (episode / "final.jpg").exists():
                    raise ValueError("episode has no final stopped image")
                names.append(name)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                excluded.append({"episode": name, "reason": str(error)})
        if not names:
            raise ValueError("No eligible stop-label episodes in the {} split".format(phase))
        feature, label, episodes = extract_split(
            model, args.dataset, names, args.device, args.chunk_size,
            args.max_terminal_neutral, args.final_repeats)
        extracted[phase] = (feature, label, episodes)
        label_audit["splits"][phase] = {
            "episodes": episodes, "samples": len(label), "positive": int(label.sum().item()),
            "negative": int((1 - label).sum().item()),
            "positive_fraction": float(label.mean().item()), "excluded": excluded}
    save_json(args.output / "label_audit.json", label_audit)
    if any(not torch.equal(original_state[name], model.state_dict()[name].detach().cpu())
           for name in original_state):
        raise RuntimeError("Frozen behavior-cloning policy changed during feature extraction")

    head = StopHead().to(args.device)
    train_features, train_labels, _ = extracted["train"]
    val_features, val_labels, _ = extracted["val"]
    train_loader = DataLoader(TensorDataset(train_features, train_labels),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_features, val_labels),
                            batch_size=args.batch_size, shuffle=False)
    positives = float(train_labels.sum().item())
    negatives = float(len(train_labels) - positives)
    pos_weight = negatives / positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=args.device))
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate)
    history, best_loss, stale, best_state = [], float("inf"), 0, None
    for epoch in range(1, args.epochs + 1):
        metrics = {"epoch": epoch}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            head.train(phase == "train")
            total = 0.
            for feature, label in loader:
                feature, label = feature.to(args.device), label.to(args.device)
                with torch.set_grad_enabled(phase == "train"):
                    loss = criterion(head(feature), label)
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                total += float(loss.item()) * len(label)
            metrics[phase + "_loss"] = total / len(loader.dataset)
        history.append(metrics)
        print(metrics, flush=True)
        if metrics["val_loss"] < best_loss - 1e-6:
            best_loss, stale = metrics["val_loss"], 0
            best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    head.load_state_dict(best_state)
    save_json(args.output / "history.json", history)

    minimum_frames = max(1, int(round(args.minimum_elapsed_seconds * saved["config"]["control"]["hz"])))
    val_probs, candidates, selected = evaluate_thresholds(
        head, *extracted["val"], args.device, args.batch_size,
        args.consecutive_frames, minimum_frames)
    threshold = selected["threshold"]
    test_features, test_labels, test_episodes = extracted["test"]
    test_probs = probabilities(head, test_features, args.device, args.batch_size)
    test_frame = frame_metrics(test_probs, test_labels.numpy(), threshold)
    test_episode = episode_metrics(test_probs, test_labels.numpy(), test_episodes, threshold,
                                   args.consecutive_frames, minimum_frames)
    decision = {"threshold": threshold, "consecutive_frames": args.consecutive_frames,
                "minimum_elapsed_s": args.minimum_elapsed_seconds}
    report = {"architecture": architecture, "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": file_sha256(args.checkpoint), "frozen_base_policy": True,
              "stop_head": "gru64-mlp32-logit1", "trainable_parameters": sum(
                  parameter.numel() for parameter in head.parameters()),
              "epochs_completed": len(history), "best_validation_loss": best_loss,
              "positive_weight": pos_weight, "decision": decision,
              "validation_selected": selected, "validation_threshold_candidates": candidates,
              "test": {"frame": test_frame, "episode": test_episode},
              "elapsed_seconds": time.time() - started}
    save_json(args.output / "metrics.json", report)
    payload = {"schema_version": 1, "artifact_type": "recurrent_stop_head",
               "architecture": architecture, "checkpoint_sha256": file_sha256(args.checkpoint),
               "state_dict": best_state, "decision": decision,
               "label_rule": label_audit["rule"], "metrics": {
                   "best_validation_loss": best_loss, "test_frame": test_frame,
                   "test_episode": {k: v for k, v in test_episode.items() if k != "details"}}}
    temporary = args.output / "stop_head.tmp"
    torch.save(payload, str(temporary))
    temporary.replace(args.output / "stop_head.pt")

    import matplotlib.pyplot as plt
    plots = args.output / "plots"
    plots.mkdir()
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([item["epoch"] for item in history], [item["train_loss"] for item in history], label="train")
    axis.plot([item["epoch"] for item in history], [item["val_loss"] for item in history], label="validation")
    axis.set(xlabel="Epoch", ylabel="Weighted BCE loss", title="Stop-head training")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(plots / "stop_loss.png"), dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4))
    truth = test_labels.numpy() >= .5
    axis.hist(test_probs[~truth], bins=30, alpha=.7, label="continue")
    axis.hist(test_probs[truth], bins=30, alpha=.7, label="terminal stop")
    axis.axvline(threshold, color="black", linestyle="--", label="threshold {:.2f}".format(threshold))
    axis.set(xlabel="Predicted stop probability", ylabel="Frames", title="Held-out stop probabilities")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(plots / "test_stop_probabilities.png"), dpi=160)
    plt.close(figure)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
