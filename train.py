"""Shared training loop and command-line entry point for all baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from eval import evaluate, forward_graph_batch
from utils import EarlyStopping, load_checkpoint, save_checkpoint


MODEL_CHOICES = (
    "mlp", "cnn", "lstm", "tcn", "gcn", "sage", "gat", "rgcn", "hgt",
    "hgt_fixed_quality", "hgt_learnable_quality", "hgt_consistency", "qc_hgt",
)


def _set_dataset_epoch(dataset: Any, epoch: int) -> None:
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    config: Mapping[str, Any],
    checkpoint_path: Path,
    logger: Any,
) -> dict[str, Any]:
    training = config["training"]
    if training["optimizer"] != "adamw":
        raise ValueError("The shared trainer currently supports optimizer=adamw.")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    if training["loss"] != "cross_entropy":
        raise ValueError("The shared trainer currently supports loss=cross_entropy.")
    criterion = nn.CrossEntropyLoss()
    stopping = EarlyStopping(
        patience=int(training["early_stopping"]["patience"]),
        min_delta=float(training["early_stopping"]["min_delta"]),
        mode=str(training["early_stopping"]["mode"]),
    )
    monitor = str(training["early_stopping"]["metric"])
    history: list[dict[str, Any]] = []
    model.to(device)
    for epoch in range(int(training["epochs"])):
        _set_dataset_epoch(train_loader.dataset, epoch)
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            target = batch.y.reshape(-1).long()
            optimizer.zero_grad(set_to_none=True)
            logits, joint_loss, _ = forward_graph_batch(model, batch)
            loss = joint_loss if joint_loss is not None else criterion(logits, target)
            loss.backward()
            clip = training["gradient_clip_norm"]
            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            optimizer.step()
            total_loss += float(loss.item()) * target.shape[0]
            total_examples += target.shape[0]
        train_loss = total_loss / total_examples
        validation = evaluate(
            model,
            validation_loader,
            device,
            num_classes,
            criterion,
            metric_config=config["evaluation"]["metrics"],
        )
        row = {"epoch": epoch + 1, "train_loss": train_loss, "validation": validation}
        history.append(row)
        logger.info(
            "epoch=%d train_loss=%.6f val_loss=%.6f val_accuracy=%.6f val_macro_f1=%.6f",
            epoch + 1, train_loss, validation["loss"], validation["accuracy"], validation["macro_f1"],
        )
        improved, should_stop = stopping.update(float(validation[monitor]))
        if improved:
            save_checkpoint(checkpoint_path, model, optimizer, epoch + 1, validation)
        if should_stop:
            logger.info("early_stopping epoch=%d best_%s=%.6f", epoch + 1, monitor, stopping.best)
            break
    best = load_checkpoint(checkpoint_path, model, device)
    return {"history": history, "best_epoch": int(best["epoch"]), "best_validation": best["metrics"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a TEP fault-diagnosis model.")
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    return parser


def cli() -> None:
    arguments = build_parser().parse_args()
    from main import run_experiment
    run_experiment(arguments.model, arguments.config, arguments.seed)


if __name__ == "__main__":
    cli()


# Design note (paper Sec. 4.8, 5.1): ordinary baselines optimize L_cls;
# quality/consistency variants optimize their YAML-weighted joint loss, while
# checkpoint selection remains validation Macro-F1 on run-disjoint data.
