"""Train and compare linear/MLP repairability predictors from an Outcome Bank."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.repairability_outcome_bank import (
    canonical_fingerprint,
    read_outcome_bank,
    summarize_outcome_bank,
)
from model.repairability_predictor import (
    FEATURE_NAMES,
    REPAIRABILITY_CHECKPOINT_SCHEMA,
    VIOLATION_FAMILIES,
    RepairabilityPredictor,
    feature_vector,
)


@dataclass(frozen=True)
class Example:
    evaluation_case_id: str
    group_id: str
    candidate_id: str
    source_recording_id: Optional[str]
    source_performer_id: Optional[str]
    features: np.ndarray
    safe_rate: float
    mean_post_risk: float
    violation_rates: np.ndarray
    seed_count: int


def aggregate_examples(bank_path: str | Path) -> tuple[list[Example], Dict[str, Any]]:
    records = read_outcome_bank(bank_path)
    if not records:
        raise ValueError("Outcome Bank is empty")
    grouped: Dict[tuple[str, str], list[Any]] = {}
    for record in records:
        grouped.setdefault((record.evaluation_case_id, record.candidate_id), []).append(
            record
        )
    examples: list[Example] = []
    for (case_id, candidate_id), rows in sorted(grouped.items()):
        vectors = np.stack([feature_vector(row.features) for row in rows])
        if not np.allclose(vectors, vectors[:1], rtol=1.0e-6, atol=1.0e-7):
            raise ValueError(
                "pre-generation features changed across seeds for "
                f"case={case_id} candidate={candidate_id}"
            )
        group_ids = {row.group_id for row in rows}
        if len(group_ids) != 1:
            raise ValueError("candidate seed group mixes split groups")
        recording_ids = {row.source_recording_id for row in rows}
        performer_ids = {row.source_performer_id for row in rows}
        if len(recording_ids) != 1 or len(performer_ids) != 1:
            raise ValueError("candidate seed group mixes source provenance")
        violations = np.zeros((len(rows), len(VIOLATION_FAMILIES)), dtype=np.float32)
        for row_index, row in enumerate(rows):
            present = set(row.violation_families)
            violations[row_index] = np.asarray(
                [float(name in present) for name in VIOLATION_FAMILIES],
                dtype=np.float32,
            )
        examples.append(
            Example(
                evaluation_case_id=case_id,
                group_id=next(iter(group_ids)),
                candidate_id=candidate_id,
                source_recording_id=next(iter(recording_ids)),
                source_performer_id=next(iter(performer_ids)),
                features=vectors[0].astype(np.float32),
                safe_rate=float(np.mean([float(row.post_safe) for row in rows])),
                mean_post_risk=float(np.mean([float(row.post_risk) for row in rows])),
                violation_rates=violations.mean(axis=0),
                seed_count=len(rows),
            )
        )
    return examples, summarize_outcome_bank(records)


def grouped_split(
    examples: Sequence[Example], *, seed: int, isolation: str
) -> Dict[str, list[Example]]:
    if isolation not in {"sequence", "recording", "performer", "all"}:
        raise ValueError("unsupported repairability split isolation")
    if isolation in {"recording", "all"} and any(
        not example.source_recording_id for example in examples
    ):
        raise ValueError(
            f"{isolation} isolation requires source_recording_id for every example"
        )
    if isolation in {"performer", "all"} and any(
        not example.source_performer_id for example in examples
    ):
        raise ValueError(
            f"{isolation} isolation requires source_performer_id for every example"
        )
    groups = sorted({example.group_id for example in examples})
    parent = {group: group for group in groups}

    def find(group: str) -> str:
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    provenance_owner: Dict[tuple[str, str], str] = {}
    for example in examples:
        for kind, value in (
            ("recording", example.source_recording_id),
            ("performer", example.source_performer_id),
        ):
            if isolation == "sequence":
                continue
            if isolation != "all" and isolation != kind:
                continue
            if not value:
                continue
            token = (kind, value)
            if token in provenance_owner:
                union(example.group_id, provenance_owner[token])
            else:
                provenance_owner[token] = example.group_id
    components: Dict[str, set[str]] = {}
    for group in groups:
        components.setdefault(find(group), set()).add(group)
    split_units = list(components.values())
    if len(split_units) < 3:
        raise ValueError(
            f"repairability training requires at least three {isolation}-disjoint components"
        )
    rng = random.Random(int(seed))
    rng.shuffle(split_units)
    test_count = max(1, int(round(0.15 * len(split_units))))
    validation_count = max(1, int(round(0.15 * len(split_units))))
    if test_count + validation_count >= len(split_units):
        test_count = 1
        validation_count = 1
    test_groups = set().union(*split_units[:test_count])
    validation_groups = set().union(
        *split_units[test_count : test_count + validation_count]
    )
    train_groups = set(groups) - test_groups - validation_groups
    split = {
        "train": [item for item in examples if item.group_id in train_groups],
        "validation": [
            item for item in examples if item.group_id in validation_groups
        ],
        "test": [item for item in examples if item.group_id in test_groups],
    }
    if any(not rows for rows in split.values()):
        raise ValueError("grouped repairability split produced an empty partition")
    for split_name, rows in split.items():
        classes = {row.safe_rate >= 0.5 for row in rows}
        if classes != {False, True}:
            raise ValueError(
                f"repairability {split_name} split lacks both safe and unsafe outcomes"
            )
    return split


def arrays(rows: Sequence[Example]) -> tuple[np.ndarray, ...]:
    return (
        np.stack([row.features for row in rows]).astype(np.float32),
        np.asarray([row.safe_rate for row in rows], dtype=np.float32),
        np.asarray([row.mean_post_risk for row in rows], dtype=np.float32),
        np.stack([row.violation_rates for row in rows]).astype(np.float32),
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = float(labels.sum())
    if positives <= 0.0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / positives)


def calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    total = max(1, len(labels))
    result = 0.0
    for index in range(int(bins)):
        low = index / float(bins)
        high = (index + 1) / float(bins)
        mask = (probabilities >= low) & (
            probabilities <= high if index == bins - 1 else probabilities < high
        )
        if np.any(mask):
            result += float(mask.sum()) / total * abs(
                float(probabilities[mask].mean()) - float(labels[mask].mean())
            )
    return float(result)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in np.geomspace(0.25, 4.0, 81):
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -60, 60)))
        loss = -np.mean(
            labels * np.log(np.maximum(probabilities, 1.0e-8))
            + (1.0 - labels) * np.log(np.maximum(1.0 - probabilities, 1.0e-8))
        )
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_temperature = float(temperature)
    return best_temperature


def model_outputs(
    model: RepairabilityPredictor,
    x: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(x).to(device))
    safe_logits = output["safe_logit"].detach().cpu().numpy()
    risks = output["post_risk_normalized"].detach().cpu().numpy()
    violations = output["violation_logits"].detach().cpu().numpy()
    return safe_logits, risks, violations


def ranking_metrics(
    rows: Sequence[Example],
    scores: np.ndarray,
    *,
    safe_probabilities: Optional[np.ndarray] = None,
    minimum_probability_spread: float = 0.0,
) -> Dict[str, float]:
    by_case: Dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        # The authoritative pre-risk gate is never bypassed by the learned ranker.
        if row.features[FEATURE_NAMES.index("pre_safe")] >= 0.5:
            by_case.setdefault(row.evaluation_case_id, []).append(index)
    chosen: list[int] = []
    abstained = 0
    rank_index = FEATURE_NAMES.index("original_rank_fraction")
    for indices in by_case.values():
        must_abstain = len(indices) < 2
        if safe_probabilities is not None and not must_abstain:
            values = [float(safe_probabilities[index]) for index in indices]
            must_abstain = (
                max(values) - min(values) < float(minimum_probability_spread)
            )
        if safe_probabilities is not None and must_abstain:
            abstained += 1
            chosen.append(min(indices, key=lambda index: rows[index].features[rank_index]))
        else:
            chosen.append(max(indices, key=lambda index: float(scores[index])))
    if not chosen:
        return {
            "top1_post_safe": float("nan"),
            "top1_post_risk": float("nan"),
            "top1_router_probability": float("nan"),
            "top1_original_rank_fraction": float("nan"),
            "top1_regret": float("nan"),
            "eligible_boundaries": 0,
            "abstention_rate": float("nan"),
        }
    safe = np.asarray([rows[index].safe_rate for index in chosen])
    risk = np.asarray([rows[index].mean_post_risk for index in chosen])
    router_index = FEATURE_NAMES.index("router_probability")
    regret = []
    for case_id, selected_index in zip(by_case, chosen):
        best = max(rows[index].safe_rate for index in by_case[case_id])
        regret.append(best - rows[selected_index].safe_rate)
    return {
        "top1_post_safe": float(safe.mean()),
        "top1_post_risk": float(risk.mean()),
        "top1_router_probability": float(
            np.mean([rows[index].features[router_index] for index in chosen])
        ),
        "top1_original_rank_fraction": float(
            np.mean([rows[index].features[rank_index] for index in chosen])
        ),
        "top1_regret": float(np.mean(regret)),
        "eligible_boundaries": len(chosen),
        "abstention_rate": float(abstained / max(1, len(chosen))),
    }


def evaluate_predictions(
    rows: Sequence[Example],
    *,
    safe_logits: np.ndarray,
    risk_normalized: np.ndarray,
    risk_mean: float,
    risk_std: float,
    temperature: float,
    risk_weight: float,
    minimum_probability_spread: float,
) -> Dict[str, Any]:
    labels = np.asarray([row.safe_rate for row in rows], dtype=np.float64)
    probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(safe_logits / float(temperature), -60, 60))
    )
    predicted_risk = np.maximum(
        0.0,
        np.expm1(np.minimum(40.0, risk_mean + risk_std * risk_normalized)),
    )
    utility = np.log(np.maximum(probabilities, 1.0e-8)) - float(risk_weight) * np.log1p(predicted_risk)
    nll = -np.mean(
        labels * np.log(np.maximum(probabilities, 1.0e-8))
        + (1.0 - labels) * np.log(np.maximum(1.0 - probabilities, 1.0e-8))
    )
    result: Dict[str, Any] = {
        "auprc": average_precision((labels >= 0.5).astype(np.float64), probabilities),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "nll": float(nll),
        "ece_10": calibration_error(labels, probabilities, bins=10),
        "post_risk_mae": float(
            np.mean(
                np.abs(
                    predicted_risk
                    - np.asarray([row.mean_post_risk for row in rows])
                )
            )
        ),
    }
    result.update(
        ranking_metrics(
            rows,
            utility,
            safe_probabilities=probabilities,
            minimum_probability_spread=minimum_probability_spread,
        )
    )
    return result


def train_one(
    architecture: str,
    split: Mapping[str, Sequence[Example]],
    *,
    output_path: Path,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    risk_weight: float,
    minimum_probability_spread: float,
    training_fingerprint: str,
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    train_x, train_y, train_risk, train_violations = arrays(split["train"])
    validation_x, validation_y, validation_risk_values, _ = arrays(split["validation"])
    feature_mean = train_x.mean(axis=0)
    feature_std = np.maximum(train_x.std(axis=0), 1.0e-6)
    train_risk_transformed = np.log1p(np.maximum(train_risk, 0.0))
    risk_mean = float(train_risk_transformed.mean())
    risk_std = max(float(train_risk_transformed.std()), 1.0e-6)

    def normalize_x(value: np.ndarray) -> np.ndarray:
        return ((value - feature_mean) / feature_std).astype(np.float32)

    model = RepairabilityPredictor(
        architecture=architecture,
        hidden_dims=(64, 64, 32),
        dropout=0.10,
        violation_dim=len(VIOLATION_FAMILIES),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1.0e-4)
    x_tensor = torch.from_numpy(normalize_x(train_x)).to(device)
    safe_tensor = torch.from_numpy(train_y).to(device)
    risk_tensor = torch.from_numpy(
        ((train_risk_transformed - risk_mean) / risk_std).astype(np.float32)
    ).to(device)
    violation_tensor = torch.from_numpy(train_violations).to(device)
    positive = max(1.0e-6, float(train_y.sum()))
    negative = max(1.0e-6, float(len(train_y) - train_y.sum()))
    pos_weight = torch.tensor(negative / positive, device=device)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_validation = float("inf")
    stale = 0
    history: list[Dict[str, float]] = []
    for epoch in range(max(1, int(epochs))):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(x_tensor)
        safe_loss = F.binary_cross_entropy_with_logits(
            output["safe_logit"], safe_tensor, pos_weight=pos_weight
        )
        risk_loss = F.smooth_l1_loss(output["post_risk_normalized"], risk_tensor)
        violation_loss = F.binary_cross_entropy_with_logits(
            output["violation_logits"], violation_tensor
        )
        loss = safe_loss + 0.35 * risk_loss + 0.20 * violation_loss
        if not torch.isfinite(loss):
            raise RuntimeError("repairability training produced non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        validation_logits, validation_risk, _ = model_outputs(
            model, normalize_x(validation_x), device=device
        )
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(validation_logits, -60, 60)))
        validation_loss = float(np.mean((probabilities - validation_y) ** 2))
        validation_risk_target = (
            np.log1p(np.maximum(validation_risk_values, 0.0)) - risk_mean
        ) / risk_std
        validation_loss += 0.05 * float(
            np.mean(np.abs(validation_risk - validation_risk_target))
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(loss.detach().cpu().item()),
                "validation_objective": validation_loss,
            }
        )
        if validation_loss + 1.0e-8 < best_validation:
            best_validation = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= max(1, int(patience)):
            break
    if best_state is None:
        raise RuntimeError("repairability training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    validation_logits, _, _ = model_outputs(
        model, normalize_x(validation_x), device=device
    )
    temperature = fit_temperature(validation_logits, validation_y)
    metrics: Dict[str, Any] = {}
    for split_name, rows in split.items():
        split_x, _, _, _ = arrays(rows)
        logits, risks, _ = model_outputs(model, normalize_x(split_x), device=device)
        metrics[split_name] = evaluate_predictions(
            rows,
            safe_logits=logits,
            risk_normalized=risks,
            risk_mean=risk_mean,
            risk_std=risk_std,
            temperature=temperature,
            risk_weight=risk_weight,
            minimum_probability_spread=minimum_probability_spread,
        )
    checkpoint = {
        "schema": REPAIRABILITY_CHECKPOINT_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "violation_families": list(VIOLATION_FAMILIES),
        "model_config": {
            "architecture": architecture,
            "hidden_dims": [64, 64, 32],
            "dropout": 0.10,
        },
        "state_dict": best_state,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "post_risk_mean": risk_mean,
        "post_risk_std": risk_std,
        "post_risk_transform": "log1p",
        "calibration_temperature": temperature,
        "selection": {
            "risk_weight": float(risk_weight),
            "minimum_probability_spread": float(minimum_probability_spread),
            "maximum_abs_z": 8.0,
            "pre_safe_gate_is_authoritative": True,
            "guard_is_authoritative": True,
        },
        "training_fingerprint": training_fingerprint,
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return {
        "checkpoint": str(output_path),
        "architecture": architecture,
        "epochs_completed": len(history),
        "best_validation_objective": best_validation,
        "calibration_temperature": temperature,
        "metrics": metrics,
        "history": history,
    }


def baseline_report(rows: Sequence[Example]) -> Dict[str, Any]:
    router_index = FEATURE_NAMES.index("router_probability")
    risk_index = FEATURE_NAMES.index("pre_risk_score")
    router_scores = np.asarray([row.features[router_index] for row in rows])
    risk_scores = -np.asarray([row.features[risk_index] for row in rows])
    return {
        "router": ranking_metrics(rows, router_scores),
        "pre_risk": ranking_metrics(rows, risk_scores),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train repairability baselines and MLP")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--risk-weight", type=float, default=0.25)
    parser.add_argument("--minimum-probability-spread", type=float, default=0.02)
    parser.add_argument("--minimum-mlp-top1-gain", type=float, default=0.01)
    parser.add_argument("--minimum-heterogeneous-boundaries", type=int, default=1)
    parser.add_argument(
        "--split-isolation",
        choices=["sequence", "recording", "performer", "all"],
        default="all",
    )
    args = parser.parse_args(argv)

    examples, bank_summary = aggregate_examples(args.bank)
    if int(bank_summary["outcome_heterogeneous_boundaries"]) < int(
        args.minimum_heterogeneous_boundaries
    ):
        raise ValueError(
            "Outcome Bank does not establish within-pool outcome heterogeneity"
        )
    split = grouped_split(
        examples, seed=args.seed, isolation=args.split_isolation
    )
    pre_safe_index = FEATURE_NAMES.index("pre_safe")
    for split_name, rows in split.items():
        if not any(row.features[pre_safe_index] >= 0.5 for row in rows):
            raise ValueError(
                f"repairability {split_name} split has no pre-safe ranking candidates"
            )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_fingerprint = canonical_fingerprint(
        {
            "bank_fingerprint": bank_summary["fingerprint"],
            "split": {
                name: sorted({row.group_id for row in rows})
                for name, rows in split.items()
            },
            "seed": int(args.seed),
            "split_isolation": args.split_isolation,
        }
    )
    device = torch.device(args.device)
    models: Dict[str, Any] = {}
    for architecture in ("linear", "mlp"):
        models[architecture] = train_one(
            architecture,
            split,
            output_path=output_dir / f"repairability_{architecture}.pt",
            device=device,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            risk_weight=args.risk_weight,
            minimum_probability_spread=args.minimum_probability_spread,
            training_fingerprint=training_fingerprint,
        )
    baselines = {name: baseline_report(rows) for name, rows in split.items()}
    linear_test = models["linear"]["metrics"]["test"]
    mlp_test = models["mlp"]["metrics"]["test"]
    mlp_top1_gain = float(mlp_test["top1_post_safe"] - linear_test["top1_post_safe"])
    strongest_baseline_safe = max(
        float(linear_test["top1_post_safe"]),
        float(baselines["test"]["router"]["top1_post_safe"]),
        float(baselines["test"]["pre_risk"]["top1_post_safe"]),
    )
    gain_over_strongest_baseline = float(
        mlp_test["top1_post_safe"] - strongest_baseline_safe
    )
    mlp_retained = bool(
        mlp_top1_gain >= float(args.minimum_mlp_top1_gain)
        and gain_over_strongest_baseline >= 0.0
        and mlp_test["top1_post_risk"] <= linear_test["top1_post_risk"] + 1.0e-8
        and mlp_test["top1_router_probability"]
        >= linear_test["top1_router_probability"] - 0.02
    )
    for architecture in ("linear", "mlp"):
        checkpoint_path = Path(models[architecture]["checkpoint"])
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint["promotion"] = {
            "rank_authorized": bool(mlp_retained if architecture == "mlp" else False),
            "gate": "mlp_vs_linear_top1_safety_risk_router_v1",
        }
        torch.save(checkpoint, checkpoint_path)
    report = {
        "schema": "repairability_training_report_v1",
        "training_fingerprint": training_fingerprint,
        "bank": bank_summary,
        "split_sizes": {name: len(rows) for name, rows in split.items()},
        "split_isolation": args.split_isolation,
        "split_groups": {
            name: sorted({row.group_id for row in rows})
            for name, rows in split.items()
        },
        "baselines": baselines,
        "models": models,
        "gate": {
            "mlp_top1_gain_over_linear": mlp_top1_gain,
            "mlp_top1_gain_over_strongest_baseline": gain_over_strongest_baseline,
            "minimum_required_gain": float(args.minimum_mlp_top1_gain),
            "mlp_retained": mlp_retained,
            "reason": (
                "nonlinear interaction supported"
                if mlp_retained
                else "linear model is preferred until the preregistered gate passes"
            ),
        },
    }
    report["report_fingerprint"] = canonical_fingerprint(report)
    report_path = output_dir / "repairability_training_report.json"
    report_path.write_text(
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
