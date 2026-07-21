"""Build and train a rate-specific monotonic DurationPredictor.

The project Event-DB is canonical column-concatenated Rot6D.  The historical
DurationPredictor architecture is intentionally retained in its native
PyTorch3D row layout, so conversion happens once while building this dataset
and is recorded in both the dataset and checkpoint contracts.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model.duration_predictor import DurationPredictor, NATIVE_ROT6D_LAYOUT
from motion_geometry.rotations import (
    CANONICAL_ROT6D_LAYOUT,
    ROT6D_LAYOUT_PYTORCH3D_ROW,
    convert_motion_rot6d_layout_np,
)
from scheduling.duration_features import (
    build_v23_condition,
    duration_bin_ids,
    inverse_time_map,
    make_fast_turn_corruption_v2,
    make_soft_event_mask,
    parse_duration_bins,
)
from scheduling.index_io import (
    load_shared_index,
    resolve_event_motion_path,
)
from support.common import load_motion
from support.scheduler_checkpoint_contracts import scheduler_training_contract


def _fixed_window(motion: np.ndarray, window_len: int) -> tuple[np.ndarray, int, int]:
    motion = np.asarray(motion, dtype=np.float32)
    if len(motion) > window_len:
        raise RuntimeError(
            f"Event has {len(motion)} frames but Duration window has only "
            f"{window_len}; silently compressing the supervision target is forbidden"
        )
    left = max(0, (window_len - len(motion)) // 2)
    right = window_len - left - len(motion)
    window = np.concatenate(
        [
            np.repeat(motion[:1], left, axis=0),
            motion,
            np.repeat(motion[-1:], right, axis=0),
        ],
        axis=0,
    )
    return window.astype(np.float32), left, left + len(motion) - 1


def _duration_edges_from_events(
    lengths: np.ndarray,
    *,
    minimum_frames: int,
    window_len: int,
) -> np.ndarray:
    valid = np.asarray(lengths, dtype=np.int32)
    valid = valid[valid >= int(minimum_frames)]
    if not len(valid):
        raise RuntimeError("No events remain for Duration-bin calibration")
    quantiles = np.quantile(valid, [0.0, 0.15, 0.35, 0.55, 0.75, 0.90, 1.0])
    scaled = np.rint(quantiles).astype(np.int32)
    scaled[0] = max(2, int(scaled[0]))
    # Edges are inclusive lower bounds; max_length + 1 keeps the maximum event
    # inside the final bin and matches DurationPredictor.duration_max_frames.
    scaled[-1] = min(int(window_len) + 1, int(valid.max()) + 1)
    scaled = np.unique(scaled)
    if len(scaled) < 3 or scaled[-1] <= scaled[0]:
        raise RuntimeError(
            f"Cannot form duration bins from lengths={valid.tolist()}"
        )
    return scaled


def build_dataset(args: argparse.Namespace) -> int:
    if args.fps <= 0.0 or args.frame_parameters_fps <= 0.0:
        raise ValueError("fps and frame_parameters_fps must be positive")
    metadata, arrays, items = load_shared_index(args.index_json, args.index_npz)
    try:
        rates = [float(value) for value in metadata.get("canonical_fps_values", [])]
        if rates != [float(args.fps)]:
            raise RuntimeError(
                f"Duration dataset FPS mismatch: index={rates}, requested={[float(args.fps)]}"
            )
        lengths = np.asarray(arrays["length"], dtype=np.int32)
    finally:
        arrays.close()
    frame_scale = float(args.fps) / float(args.frame_parameters_fps)
    min_event_frames = max(2, int(round(args.min_event_frames * frame_scale)))
    min_corrupted_duration = max(
        3, int(round(args.min_corrupted_duration * frame_scale))
    )
    mask_context = max(1, int(round(args.mask_context * frame_scale)))
    window_len = int(args.window_len)
    if window_len <= 0:
        window_len = max(
            int(round(120.0 * frame_scale)),
            int(lengths.max()) + 2 * mask_context,
        )
    if int(lengths.max()) > window_len:
        raise RuntimeError(
            f"Duration window_len={window_len} is smaller than the longest "
            f"Generation event ({int(lengths.max())} frames)"
        )
    edges = (
        parse_duration_bins(args.duration_edges)
        if args.duration_edges
        else _duration_edges_from_events(
            lengths,
            minimum_frames=min_event_frames,
            window_len=window_len,
        )
    )
    if edges[-1] > window_len + 1:
        raise RuntimeError(
            f"Duration edge {edges[-1]} exceeds window_len={window_len}"
        )
    eligible_lengths = lengths[lengths >= min_event_frames]
    if len(edges) < 3 or not len(eligible_lengths):
        raise RuntimeError("Duration training requires at least two calibrated bins")
    if int(edges[0]) > int(eligible_lengths.min()) or int(edges[-1]) <= int(
        eligible_lengths.max()
    ):
        raise RuntimeError(
            "Duration edges do not cover every eligible event: "
            f"edges=[{int(edges[0])}, {int(edges[-1])}], "
            f"events=[{int(eligible_lengths.min())}, {int(eligible_lengths.max())}]"
        )

    rng = np.random.default_rng(args.seed)
    records: dict[str, list[Any]] = {
        key: []
        for key in (
            "corrupted",
            "edit_mask",
            "condition",
            "target_tau",
            "target_duration_frames",
            "is_identity",
            "duration_bin",
            "event_uids",
            "source_uids",
            "speed_factor",
        )
    }
    for event_index, item in enumerate(items):
        path = resolve_event_motion_path(item, args.index_json, metadata=metadata)
        motion = load_motion(path)
        if len(motion) < min_event_frames:
            continue
        target_duration = len(motion)
        target_window, event_start, event_end = _fixed_window(motion, window_len)
        augmentation_count = max(1, args.augmentations_per_event)
        factors = [1.0]
        if augmentation_count > 1:
            factors.extend(
                rng.uniform(args.min_speed_factor, args.max_speed_factor, augmentation_count - 1).tolist()
            )
        for factor in factors:
            if factor <= 1.0 + 1.0e-6:
                corrupted = target_window.copy()
                edit_mask = make_soft_event_mask(
                    window_len, event_start, event_end, context=mask_context
                )
                target_tau = np.linspace(0.0, 1.0, window_len, dtype=np.float32)
                corrupted_start, corrupted_end = event_start, event_end
                identity = 1.0
            else:
                corrupted, edit_mask, info = make_fast_turn_corruption_v2(
                    target_window,
                    event_start,
                    event_end,
                    speed_factor=float(factor),
                    min_context_frames=max(4, mask_context),
                    min_corrupted_duration=min_corrupted_duration,
                    max_effective_factor=args.max_speed_factor,
                )
                target_tau = inverse_time_map(np.asarray(info["source_positions"], dtype=np.float32))
                corrupted_start = int(info["corrupted_turn_start"])
                corrupted_end = int(info["corrupted_turn_end"])
                identity = 0.0
            condition = build_v23_condition(
                corrupted,
                corrupted_start,
                corrupted_end,
                fps=args.fps,
                rot6d_layout=CANONICAL_ROT6D_LAYOUT,
            )
            corrupted_native = convert_motion_rot6d_layout_np(
                corrupted,
                CANONICAL_ROT6D_LAYOUT,
                ROT6D_LAYOUT_PYTORCH3D_ROW,
            )
            records["corrupted"].append(corrupted_native)
            records["edit_mask"].append(edit_mask.astype(np.float32))
            records["condition"].append(condition.astype(np.float32))
            records["target_tau"].append(target_tau.astype(np.float32))
            records["target_duration_frames"].append(float(target_duration))
            records["is_identity"].append(identity)
            records["duration_bin"].append(
                int(duration_bin_ids(np.asarray([target_duration]), edges)[0])
            )
            records["event_uids"].append(str(item["event_uid"]))
            records["source_uids"].append(str(item.get("source_uid", "unknown")))
            records["speed_factor"].append(float(factor))
        if (event_index + 1) % 100 == 0 or event_index + 1 == len(items):
            print(f"[Duration data] {event_index + 1}/{len(items)} events", flush=True)

    if not records["corrupted"]:
        raise RuntimeError("Duration training dataset is empty")
    target = Path(args.out).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        corrupted=np.stack(records["corrupted"]).astype(np.float32),
        edit_mask=np.stack(records["edit_mask"]).astype(np.float32),
        condition=np.stack(records["condition"]).astype(np.float32),
        target_tau=np.stack(records["target_tau"]).astype(np.float32),
        target_duration_frames=np.asarray(records["target_duration_frames"], dtype=np.float32),
        is_identity=np.asarray(records["is_identity"], dtype=np.float32),
        duration_bin=np.asarray(records["duration_bin"], dtype=np.int64),
        event_uids=np.asarray(records["event_uids"], dtype=object),
        source_uids=np.asarray(records["source_uids"], dtype=object),
        speed_factor=np.asarray(records["speed_factor"], dtype=np.float32),
        duration_edges=edges.astype(np.float32),
        fps=np.asarray(float(args.fps), dtype=np.float32),
        source_rot6d_layout=np.asarray(CANONICAL_ROT6D_LAYOUT, dtype=object),
        model_rot6d_layout=np.asarray(ROT6D_LAYOUT_PYTORCH3D_ROW, dtype=object),
        event_db_contract_json=np.asarray(
            json.dumps(metadata["event_db_contract"], sort_keys=True), dtype=object
        ),
    )
    report = {
        "schema": "dunhuang_duration_dataset_v1",
        "dataset": str(target),
        "num_samples": len(records["corrupted"]),
        "num_events": len(set(records["event_uids"])),
        "num_sources": len(set(records["source_uids"])),
        "fps": float(args.fps),
        "duration_edges": edges.tolist(),
        "window_len": window_len,
        "source_rot6d_layout": CANONICAL_ROT6D_LAYOUT,
        "model_rot6d_layout": ROT6D_LAYOUT_PYTORCH3D_ROW,
        "event_db_contract": metadata["event_db_contract"],
    }
    target.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


def _source_split(groups: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(sorted({str(value) for value in groups}), dtype=object)
    if len(unique) < 2:
        raise RuntimeError("Duration training requires at least two source groups")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    count = max(1, min(len(unique) - 1, int(round(len(unique) * val_ratio))))
    validation = {str(value) for value in unique[:count]}
    train = np.asarray([i for i, value in enumerate(groups) if str(value) not in validation], dtype=np.int64)
    val = np.asarray([i for i, value in enumerate(groups) if str(value) in validation], dtype=np.int64)
    return train, val


def train_model(args: argparse.Namespace) -> int:
    if args.fps <= 0.0 or args.frame_parameters_fps <= 0.0:
        raise ValueError("fps and frame_parameters_fps must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with np.load(args.data, allow_pickle=True) as data:
        dataset_fps = float(np.asarray(data["fps"]).item())
        model_layout = str(np.asarray(data["model_rot6d_layout"]).item())
        if abs(dataset_fps - float(args.fps)) > 1.0e-6:
            raise RuntimeError(
                f"Duration dataset FPS mismatch: data={dataset_fps}, requested={args.fps}"
            )
        if model_layout != ROT6D_LAYOUT_PYTORCH3D_ROW:
            raise RuntimeError(f"Duration dataset has wrong model layout: {model_layout!r}")
        names = (
            "corrupted",
            "edit_mask",
            "condition",
            "target_tau",
            "target_duration_frames",
            "is_identity",
            "duration_bin",
        )
        values = [np.asarray(data[name]) for name in names]
        source_uids = np.asarray(data["source_uids"], dtype=object)
        duration_edges = np.asarray(data["duration_edges"], dtype=np.float32)
    train_indices, val_indices = _source_split(source_uids, args.val_ratio, args.seed)

    tensors = [
        torch.from_numpy(value.astype(np.int64 if name == "duration_bin" else np.float32))
        for name, value in zip(names, values)
    ]

    def loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        index_tensor = torch.from_numpy(indices.astype(np.int64, copy=False))
        dataset = TensorDataset(
            *(tensor.index_select(0, index_tensor) for tensor in tensors)
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = loader(train_indices, True)
    val_loader = loader(val_indices, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slow_feature_span = max(
        1,
        int(round(args.slow_feature_span * float(args.fps) / args.frame_parameters_fps)),
    )
    model = DurationPredictor(
        motion_dim=151,
        condition_dim=values[2].shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        duration_edges=duration_edges.tolist(),
        window_len=values[0].shape[1],
        slow_feature_span=slow_feature_span,
        ordinal_blend=args.ordinal_blend,
        fps=args.fps,
    ).to(device)
    model.set_train_stage("joint")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def run_epoch(data_loader: DataLoader, training: bool) -> tuple[float, dict[str, float]]:
        model.train(training)
        totals = {key: 0.0 for key in ("loss", "duration", "ordinal", "edit", "tau")}
        count = 0
        for batch in data_loader:
            corrupted, edit_mask, condition, target_tau, target_duration, identity, target_bin = [
                value.to(device, non_blocking=True) for value in batch
            ]
            with torch.set_grad_enabled(training):
                output = model(
                    corrupted,
                    edit_mask,
                    condition,
                    duration_override_frames=target_duration,
                )
                duration_loss = F.smooth_l1_loss(
                    output["duration_soft_frames"] / float(args.fps),
                    target_duration / float(args.fps),
                )
                thresholds = torch.arange(
                    model.num_duration_bins - 1, device=device
                )[None]
                ordinal_target = (target_bin[:, None] > thresholds).to(torch.float32)
                ordinal_loss = F.binary_cross_entropy_with_logits(
                    output["duration_ordinal_logits"], ordinal_target
                )
                edit_loss = F.binary_cross_entropy_with_logits(
                    output["edit_logit"], 1.0 - identity
                )
                tau_loss = F.smooth_l1_loss(output["tau"], target_tau)
                loss = (
                    args.duration_weight * duration_loss
                    + args.ordinal_weight * ordinal_loss
                    + args.edit_weight * edit_loss
                    + args.tau_weight * tau_loss
                )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            batch_size = len(corrupted)
            for key, value in (
                ("loss", loss),
                ("duration", duration_loss),
                ("ordinal", ordinal_loss),
                ("edit", edit_loss),
                ("tau", tau_loss),
            ):
                totals[key] += float(value.detach()) * batch_size
            count += batch_size
        return totals["loss"] / max(count, 1), {
            key: value / max(count, 1) for key, value in totals.items()
        }

    metadata, arrays, _items = load_shared_index(args.index_json, args.index_npz)
    arrays.close()
    contract = scheduler_training_contract(
        role="duration",
        fps=args.fps,
        index_metadata=metadata,
        index_json=args.index_json,
        index_npz=args.index_npz,
        dataset=args.data,
        model_rot6d_layout=NATIVE_ROT6D_LAYOUT,
    )
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "motion_dim": 151,
        "condition_dim": int(values[2].shape[1]),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "duration_edges": duration_edges.tolist(),
        "window_len": int(values[0].shape[1]),
        "slow_feature_span": slow_feature_span,
        "ordinal_blend": args.ordinal_blend,
        "fps": float(args.fps),
        "rot6d_layout": NATIVE_ROT6D_LAYOUT,
    }
    best = float("inf")
    patience = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(train_loader, True)
        val_loss, val_metrics = run_epoch(val_loader, False)
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        )
        if val_loss < best - 1.0e-7:
            best = val_loss
            patience = 0
            torch.save(
                {
                    "version": "formal_monotonic_duration_v1",
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "fps": float(args.fps),
                    "rot6d_layout": NATIVE_ROT6D_LAYOUT,
                    "canonical_rot6d_layout": CANONICAL_ROT6D_LAYOUT,
                    "epoch": epoch,
                    "stage": "joint",
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "scheduler_contract": contract,
                },
                output,
            )
        else:
            patience += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[Duration] epoch={epoch} train={train_loss:.6f} "
                f"val={val_loss:.6f} best={best:.6f}",
                flush=True,
            )
        if patience >= args.patience:
            break
    output.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"[PASS] Duration checkpoint: {output}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-dataset")
    build.add_argument("--index_json", required=True)
    build.add_argument("--index_npz", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--fps", type=float, required=True)
    build.add_argument(
        "--window_len",
        type=int,
        default=0,
        help="0 selects a rate- and Event-DB-safe window automatically",
    )
    build.add_argument("--frame_parameters_fps", type=float, default=30.0)
    build.add_argument("--duration_edges", default="")
    build.add_argument("--augmentations_per_event", type=int, default=4)
    build.add_argument("--min_event_frames", type=int, default=12)
    build.add_argument("--min_speed_factor", type=float, default=1.15)
    build.add_argument("--max_speed_factor", type=float, default=2.5)
    build.add_argument("--min_corrupted_duration", type=int, default=4)
    build.add_argument("--mask_context", type=int, default=6)
    build.add_argument("--seed", type=int, default=20260722)

    train = subparsers.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--index_json", required=True)
    train.add_argument("--index_npz", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--fps", type=float, required=True)
    train.add_argument("--frame_parameters_fps", type=float, default=30.0)
    train.add_argument("--epochs", type=int, default=240)
    train.add_argument("--batch_size", type=int, default=40)
    train.add_argument("--lr", type=float, default=3.0e-5)
    train.add_argument("--weight_decay", type=float, default=1.0e-3)
    train.add_argument("--hidden_dim", type=int, default=96)
    train.add_argument("--dropout", type=float, default=0.24)
    train.add_argument("--slow_feature_span", type=int, default=10)
    train.add_argument("--ordinal_blend", type=float, default=0.82)
    train.add_argument("--duration_weight", type=float, default=1.0)
    train.add_argument("--ordinal_weight", type=float, default=1.0)
    train.add_argument("--edit_weight", type=float, default=0.25)
    train.add_argument("--tau_weight", type=float, default=1.0)
    train.add_argument("--val_ratio", type=float, default=0.20)
    train.add_argument("--patience", type=int, default=50)
    train.add_argument("--num_workers", type=int, default=4)
    train.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return build_dataset(args) if args.command == "build-dataset" else train_model(args)


if __name__ == "__main__":
    raise SystemExit(main())
