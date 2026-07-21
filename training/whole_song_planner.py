"""Build weakly supervised sequences and train the formal whole-song Planner.

The labels are produced only from the current Generation-aligned Event-DB and
the newly trained Router.  Songs, not phrases, define the train/validation
split, which prevents neighbouring phrases from leaking across the split.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from model.music_motion_router import load_router_checkpoint
from model.whole_song_planner import WholeSongPlanner
from scheduling.index_io import load_shared_index
from scheduling.music_phrase_segmentation import (
    segment_music_phrases,
    split_music_phrases_for_events,
    whole_song_features,
)
from support.common import (
    EVENT_TO_ID,
    event_compatibility,
    family_id,
    transition_cost_from_arrays,
)
from training.music_corpus import audio_sha256, discover_training_audio
from support.scheduler_checkpoint_contracts import (
    assert_scheduler_checkpoint_contract,
    scheduler_training_contract,
    sha256_file,
)


def _object_array(values: list[np.ndarray]) -> np.ndarray:
    result = np.empty((len(values),), dtype=object)
    result[:] = values
    return result


def _scaled_transition_lengths(fps: float) -> tuple[int, ...]:
    values = np.rint(
        np.asarray([12, 16, 20, 24, 30, 36, 42, 48], dtype=np.float64)
        * float(fps)
        / 30.0
    ).astype(np.int64)
    return tuple(int(value) for value in np.unique(np.maximum(values, 1)))


def _select_event_path(
    *,
    phrases: Sequence[Any],
    similarity: np.ndarray,
    items: Sequence[dict[str, Any]],
    natural_duration: np.ndarray,
    quality: np.ndarray,
    entry_pose: np.ndarray,
    exit_pose: np.ndarray,
    entry_velocity: np.ndarray,
    exit_velocity: np.ndarray,
    entry_angular_velocity: np.ndarray,
    exit_angular_velocity: np.ndarray,
    fps: float,
    cooldown_slots: int,
    candidate_pool: int,
) -> list[int]:
    event_types = [str(item.get("event_type", "neutral_flow")) for item in items]
    sources = [str(item.get("source_uid", "unknown")) for item in items]
    families = [str(item.get("family_id", family_id(item))) for item in items]
    selected: list[int] = []
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()

    for slot, phrase in enumerate(phrases):
        phrase_target = max(
            1.0,
            float(phrase.length - (0 if slot == 0 else phrase.transition_base_frames)),
        )
        compatibility = np.asarray(
            [event_compatibility(phrase.music_event, value) for value in event_types],
            dtype=np.float32,
        )
        duration_cost = np.abs(
            np.log((natural_duration + 1.0) / (phrase_target + 1.0))
        )
        score = (
            np.asarray(similarity[slot], dtype=np.float32)
            + 0.70 * compatibility
            + 0.30 * quality
            - 0.28 * duration_cost
        )
        history = max(1, len(selected))
        for event_index in range(len(items)):
            if event_index in selected[-max(0, int(cooldown_slots)) :]:
                score[event_index] -= 4.0
            score[event_index] -= 0.45 * source_counts[sources[event_index]] / history
            score[event_index] -= 0.30 * family_counts[families[event_index]] / history
        shortlist = np.argsort(score)[::-1][
            : max(1, min(int(candidate_pool), len(score)))
        ]
        shortlist_score = score[shortlist].copy()
        if selected:
            previous = selected[-1]
            boundary_cost = np.asarray(
                [
                    transition_cost_from_arrays(
                        exit_pose[previous],
                        exit_velocity[previous],
                        entry_pose[int(candidate)],
                        entry_velocity[int(candidate)],
                    )
                    for candidate in shortlist
                ],
                dtype=np.float32,
            )
            angular_cost = np.linalg.norm(
                exit_angular_velocity[previous][None]
                - entry_angular_velocity[shortlist],
                axis=-1,
            ).mean(axis=-1)
            # Match the formal runtime's preference for intrinsically smooth
            # boundaries without allowing a single noisy endpoint to dominate
            # the weak semantic label.
            shortlist_score -= 0.35 * np.minimum(boundary_cost, 3.0)
            shortlist_score -= 0.15 * np.minimum(angular_cost / 4.0, 2.0)
        chosen = int(shortlist[int(np.argmax(shortlist_score))])
        selected.append(chosen)
        source_counts[sources[chosen]] += 1
        family_counts[families[chosen]] += 1
    return selected


def build_dataset(args: argparse.Namespace) -> int:
    metadata, arrays, items = load_shared_index(args.index_json, args.index_npz)
    try:
        rates = [float(value) for value in metadata.get("canonical_fps_values", [])]
        if rates != [float(args.fps)]:
            raise RuntimeError(
                f"Planner dataset FPS mismatch: index={rates}, requested={[float(args.fps)]}"
            )
        motion_desc = np.asarray(arrays["motion_desc"], dtype=np.float32)
        natural_duration = np.asarray(arrays["natural_duration"], dtype=np.float32)
        quality = (
            0.50 * np.asarray(arrays["quality_score"], dtype=np.float32)
            + 0.30 * np.asarray(arrays["safety_score"], dtype=np.float32)
            + 0.20 * np.asarray(arrays["style_score"], dtype=np.float32)
        )
        entry_pose = np.asarray(arrays["entry_pose"], dtype=np.float32)
        exit_pose = np.asarray(arrays["exit_pose"], dtype=np.float32)
        entry_velocity = np.asarray(arrays["entry_vel"], dtype=np.float32)
        exit_velocity = np.asarray(arrays["exit_vel"], dtype=np.float32)
        entry_angular_velocity = np.asarray(
            arrays["entry_angular_velocity_radps"], dtype=np.float32
        )
        exit_angular_velocity = np.asarray(
            arrays["exit_angular_velocity_radps"], dtype=np.float32
        )
    finally:
        arrays.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router_checkpoint = torch.load(
        args.router_ckpt, map_location="cpu", weights_only=False
    )
    assert_scheduler_checkpoint_contract(
        router_checkpoint,
        role="router",
        runtime_fps=args.fps,
        event_db_contract=metadata["event_db_contract"],
        index_json=args.index_json,
        index_npz=args.index_npz,
        path=args.router_ckpt,
        allow_legacy_30fps=False,
    )
    duration_checkpoint = torch.load(
        args.duration_ckpt, map_location="cpu", weights_only=False
    )
    assert_scheduler_checkpoint_contract(
        duration_checkpoint,
        role="duration",
        runtime_fps=args.fps,
        event_db_contract=metadata["event_db_contract"],
        index_json=args.index_json,
        index_npz=args.index_npz,
        path=args.duration_ckpt,
        allow_legacy_30fps=False,
    )
    router = load_router_checkpoint(args.router_ckpt, device=device)
    with torch.no_grad():
        motion_embedding = router.encode_motion(
            torch.from_numpy(motion_desc).to(device)
        )

    audio_paths = discover_training_audio(args.music_dirs)
    cache_dir = Path(args.cache_dir).resolve()
    feature_sequences: list[np.ndarray] = []
    event_sequences: list[np.ndarray] = []
    duration_sequences: list[np.ndarray] = []
    transition_sequences: list[np.ndarray] = []
    activity_sequences: list[np.ndarray] = []
    selected_uid_sequences: list[np.ndarray] = []
    song_uids: list[str] = []
    transition_lengths = _scaled_transition_lengths(args.fps)

    for song_index, audio_path in enumerate(audio_paths):
        features, _feature_metadata = whole_song_features(
            audio_path,
            fps=args.fps,
            cache_dir=cache_dir,
            max_seconds=args.max_seconds,
        )
        phrases, _segmentation = segment_music_phrases(
            features,
            fps=args.fps,
            min_phrase_seconds=args.min_phrase_seconds,
            max_phrase_seconds=args.max_phrase_seconds,
        )
        phrases, _slots = split_music_phrases_for_events(
            features,
            phrases,
            fps=args.fps,
            enabled=True,
            max_slot_seconds=args.max_slot_seconds,
            min_slot_seconds=args.min_slot_seconds,
            max_events_per_phrase=args.max_events_per_phrase,
        )
        if not phrases:
            continue
        queries = np.stack(
            [np.asarray(phrase.query, dtype=np.float32) for phrase in phrases]
        )
        with torch.no_grad():
            music_embedding = router.encode_music(
                torch.from_numpy(queries).to(device)
            )
            similarity = (music_embedding @ motion_embedding.t()).cpu().numpy()
        chosen = _select_event_path(
            phrases=phrases,
            similarity=similarity,
            items=items,
            natural_duration=natural_duration,
            quality=quality,
            entry_pose=entry_pose,
            exit_pose=exit_pose,
            entry_velocity=entry_velocity,
            exit_velocity=exit_velocity,
            entry_angular_velocity=entry_angular_velocity,
            exit_angular_velocity=exit_angular_velocity,
            fps=args.fps,
            cooldown_slots=args.cooldown_slots,
            candidate_pool=args.weak_candidate_pool,
        )
        event_ids = []
        duration_targets = []
        selected_uids = []
        transition_targets = []
        for slot, event_index in enumerate(chosen):
            event_name = str(items[event_index].get("event_type", "neutral_flow"))
            event_ids.append(EVENT_TO_ID.get(event_name, EVENT_TO_ID["neutral_flow"]))
            duration_targets.append(float(natural_duration[event_index]))
            selected_uids.append(str(items[event_index]["event_uid"]))
            if slot == 0:
                transition_targets.append(-100)
            else:
                target = int(phrases[slot].transition_base_frames)
                transition_targets.append(
                    int(np.argmin(np.abs(np.asarray(transition_lengths) - target)))
                )
        feature_sequences.append(
            np.stack(
                [np.asarray(phrase.planner_feature, dtype=np.float32) for phrase in phrases]
            )
        )
        event_sequences.append(np.asarray(event_ids, dtype=np.int64))
        duration_sequences.append(np.asarray(duration_targets, dtype=np.float32))
        transition_sequences.append(np.asarray(transition_targets, dtype=np.int64))
        activity_sequences.append(
            np.asarray([phrase.arousal for phrase in phrases], dtype=np.float32)
        )
        selected_uid_sequences.append(np.asarray(selected_uids, dtype=object))
        song_uids.append("aud_" + audio_sha256(audio_path)[:24])
        if (song_index + 1) % 25 == 0 or song_index + 1 == len(audio_paths):
            print(f"[Planner data] {song_index + 1}/{len(audio_paths)} songs", flush=True)

    if len(feature_sequences) < 2:
        raise RuntimeError("Planner training requires at least two usable songs")
    target = Path(args.out).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        features=_object_array(feature_sequences),
        event_ids=_object_array(event_sequences),
        duration_frames=_object_array(duration_sequences),
        transition_class=_object_array(transition_sequences),
        activity=_object_array(activity_sequences),
        selected_event_uids=_object_array(selected_uid_sequences),
        song_uids=np.asarray(song_uids, dtype=object),
        transition_lengths=np.asarray(transition_lengths, dtype=np.int64),
        fps=np.asarray(float(args.fps), dtype=np.float32),
        event_db_contract_json=np.asarray(
            json.dumps(metadata["event_db_contract"], sort_keys=True), dtype=object
        ),
        router_checkpoint=np.asarray(str(Path(args.router_ckpt).resolve()), dtype=object),
        router_checkpoint_sha256=np.asarray(
            sha256_file(args.router_ckpt), dtype=object
        ),
        duration_checkpoint=np.asarray(
            str(Path(args.duration_ckpt).resolve()), dtype=object
        ),
        duration_checkpoint_sha256=np.asarray(
            sha256_file(args.duration_ckpt), dtype=object
        ),
    )
    report = {
        "schema": "dunhuang_whole_song_planner_dataset_v1",
        "supervision": "current_router_and_contract_constrained_weak_labels",
        "dataset": str(target),
        "num_songs": len(feature_sequences),
        "num_slots": int(sum(len(value) for value in feature_sequences)),
        "fps": float(args.fps),
        "transition_lengths": list(transition_lengths),
        "event_db_contract": metadata["event_db_contract"],
        "router_checkpoint": str(Path(args.router_ckpt).resolve()),
        "duration_checkpoint": str(Path(args.duration_ckpt).resolve()),
    }
    target.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


class _PlannerDataset(Dataset):
    def __init__(self, payload: dict[str, list[np.ndarray]], indices: np.ndarray) -> None:
        self.payload = payload
        self.indices = [int(value) for value in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, ...]:
        index = self.indices[position]
        return (
            torch.from_numpy(self.payload["features"][index].astype(np.float32)),
            torch.from_numpy(self.payload["event_ids"][index].astype(np.int64)),
            torch.from_numpy(self.payload["duration_frames"][index].astype(np.float32)),
            torch.from_numpy(self.payload["transition_class"][index].astype(np.int64)),
            torch.from_numpy(self.payload["activity"][index].astype(np.float32)),
        )


def _collate(batch: Sequence[tuple[torch.Tensor, ...]]) -> dict[str, torch.Tensor]:
    lengths = torch.tensor([len(row[0]) for row in batch], dtype=torch.long)
    maximum = int(lengths.max())
    positions = torch.arange(maximum)[None]
    padding_mask = positions >= lengths[:, None]
    return {
        "features": pad_sequence([row[0] for row in batch], batch_first=True),
        "event_ids": pad_sequence(
            [row[1] for row in batch], batch_first=True, padding_value=-100
        ),
        "duration_frames": pad_sequence(
            [row[2] for row in batch], batch_first=True, padding_value=0.0
        ),
        "transition_class": pad_sequence(
            [row[3] for row in batch], batch_first=True, padding_value=-100
        ),
        "activity": pad_sequence(
            [row[4] for row in batch], batch_first=True, padding_value=0.0
        ),
        "padding_mask": padding_mask,
    }


def _song_split(count: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 2:
        raise RuntimeError("Planner training requires at least two songs")
    indices = np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_count = max(1, min(count - 1, int(round(count * val_ratio))))
    return indices[val_count:], indices[:val_count]


def train_model(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with np.load(args.data, allow_pickle=True) as data:
        dataset_fps = float(np.asarray(data["fps"]).item())
        if abs(dataset_fps - float(args.fps)) > 1.0e-6:
            raise RuntimeError(
                f"Planner dataset FPS mismatch: data={dataset_fps}, requested={args.fps}"
            )
        names = (
            "features",
            "event_ids",
            "duration_frames",
            "transition_class",
            "activity",
        )
        payload = {
            name: [np.asarray(value) for value in data[name].tolist()]
            for name in names
        }
        song_uids = np.asarray(data["song_uids"], dtype=object)
        transition_lengths = tuple(
            int(value) for value in np.asarray(data["transition_lengths"])
        )
        router_path = Path(str(np.asarray(data["router_checkpoint"]).item()))
        duration_path = Path(str(np.asarray(data["duration_checkpoint"]).item()))
        router_hash = str(np.asarray(data["router_checkpoint_sha256"]).item())
        duration_hash = str(np.asarray(data["duration_checkpoint_sha256"]).item())
    if sha256_file(router_path) != router_hash:
        raise RuntimeError(
            "Router checkpoint changed after Planner weak labels were generated"
        )
    if sha256_file(duration_path) != duration_hash:
        raise RuntimeError(
            "Duration checkpoint changed after Planner dataset construction"
        )
    if len(set(str(value) for value in song_uids)) != len(song_uids):
        raise RuntimeError("Planner dataset contains duplicate song identities")
    train_indices, val_indices = _song_split(len(song_uids), args.val_ratio, args.seed)
    train_loader = DataLoader(
        _PlannerDataset(payload, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        _PlannerDataset(payload, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )
    feature_dim = int(payload["features"][0].shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WholeSongPlanner(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        transition_lengths=transition_lengths,
        fps=args.fps,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def run_epoch(loader: DataLoader, training: bool) -> tuple[float, dict[str, float]]:
        model.train(training)
        totals = {name: 0.0 for name in ("loss", "event", "duration", "transition", "activity")}
        samples = 0
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            valid = ~batch["padding_mask"]
            with torch.set_grad_enabled(training):
                output = model(batch["features"], padding_mask=batch["padding_mask"])
                event_loss = F.cross_entropy(
                    output["event_logits"].reshape(-1, output["event_logits"].shape[-1]),
                    batch["event_ids"].reshape(-1),
                    ignore_index=-100,
                )
                duration_loss = F.smooth_l1_loss(
                    output["log_duration"][valid],
                    torch.log(batch["duration_frames"][valid].clamp_min(1.0)),
                )
                transition_valid = batch["transition_class"] != -100
                if bool(transition_valid.any()):
                    transition_loss = F.cross_entropy(
                        output["transition_logits"][transition_valid],
                        batch["transition_class"][transition_valid],
                    )
                else:
                    transition_loss = output["transition_logits"].sum() * 0.0
                activity_loss = F.binary_cross_entropy(
                    output["activity"][valid], batch["activity"][valid].clamp(0.0, 1.0)
                )
                loss = (
                    args.event_weight * event_loss
                    + args.duration_weight * duration_loss
                    + args.transition_weight * transition_loss
                    + args.activity_weight * activity_loss
                )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            count = int(valid.sum())
            for name, value in (
                ("loss", loss),
                ("event", event_loss),
                ("duration", duration_loss),
                ("transition", transition_loss),
                ("activity", activity_loss),
            ):
                totals[name] += float(value.detach()) * count
            samples += count
        return totals["loss"] / max(samples, 1), {
            name: value / max(samples, 1) for name, value in totals.items()
        }

    metadata, arrays, _items = load_shared_index(args.index_json, args.index_npz)
    arrays.close()
    contract = scheduler_training_contract(
        role="planner",
        fps=args.fps,
        index_metadata=metadata,
        index_json=args.index_json,
        index_npz=args.index_npz,
        dataset=args.data,
        upstream_checkpoints={"router": router_path, "duration": duration_path},
    )
    config: dict[str, Any] = {
        "feature_dim": feature_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "num_event_types": len(EVENT_TO_ID),
        "transition_lengths": list(transition_lengths),
        "fps": float(args.fps),
        "min_duration_seconds": 8.0 / 30.0,
        "max_duration_seconds": 20.0,
    }
    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best = math.inf
    patience = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(train_loader, True)
        val_loss, val_metrics = run_epoch(val_loader, False)
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        if val_loss < best - 1.0e-7:
            best = val_loss
            patience = 0
            torch.save(
                {
                    "version": "formal_whole_song_planner_v1",
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "fps": float(args.fps),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "scheduler_contract": contract,
                },
                output_path,
            )
        else:
            patience += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[Planner] epoch={epoch} train={train_loss:.6f} "
                f"val={val_loss:.6f} best={best:.6f}",
                flush=True,
            )
        if patience >= args.patience:
            break
    output_path.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"[PASS] Planner checkpoint: {output_path}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-dataset")
    build.add_argument("--index_json", required=True)
    build.add_argument("--index_npz", required=True)
    build.add_argument("--router_ckpt", required=True)
    build.add_argument("--duration_ckpt", required=True)
    build.add_argument("--music_dirs", nargs="+", required=True)
    build.add_argument("--cache_dir", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--fps", type=float, required=True)
    build.add_argument("--max_seconds", type=float, default=0.0)
    build.add_argument("--min_phrase_seconds", type=float, default=2.5)
    build.add_argument("--max_phrase_seconds", type=float, default=7.5)
    build.add_argument("--max_slot_seconds", type=float, default=3.2)
    build.add_argument("--min_slot_seconds", type=float, default=1.6)
    build.add_argument("--max_events_per_phrase", type=int, default=4)
    build.add_argument("--cooldown_slots", type=int, default=8)
    build.add_argument("--weak_candidate_pool", type=int, default=256)

    train = subparsers.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--index_json", required=True)
    train.add_argument("--index_npz", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--fps", type=float, required=True)
    train.add_argument("--epochs", type=int, default=180)
    train.add_argument("--batch_size", type=int, default=12)
    train.add_argument("--lr", type=float, default=1.0e-4)
    train.add_argument("--weight_decay", type=float, default=1.0e-3)
    train.add_argument("--hidden_dim", type=int, default=128)
    train.add_argument("--num_layers", type=int, default=4)
    train.add_argument("--num_heads", type=int, default=4)
    train.add_argument("--dropout", type=float, default=0.15)
    train.add_argument("--event_weight", type=float, default=1.0)
    train.add_argument("--duration_weight", type=float, default=0.65)
    train.add_argument("--transition_weight", type=float, default=0.50)
    train.add_argument("--activity_weight", type=float, default=0.25)
    train.add_argument("--val_ratio", type=float, default=0.15)
    train.add_argument("--patience", type=int, default=35)
    train.add_argument("--num_workers", type=int, default=4)
    train.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return build_dataset(args) if args.command == "build-dataset" else train_model(args)


if __name__ == "__main__":
    raise SystemExit(main())
