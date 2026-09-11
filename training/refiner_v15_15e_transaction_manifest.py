"""Freeze a leakage-resistant V15.15e multi-transaction teacher manifest.

This command performs no Oracle search and consumes no generated teacher.
It deterministically fixes transaction membership and case ownership before
the expensive feasibility runs begin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from training import motion_models as m
from training import refiner_projected_candidate_probe as projected_probe


SCHEMA = "refiner_v15_15e_multi_transaction_teacher_manifest_v1"
SPLIT_SALT = "v15.15e-pre-oracle-train-validation-split-v1"


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_digest(*parts):
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_case_identity(selected, anchor_cases, local_case_index):
    local = int(local_case_index)
    block, within = divmod(local, int(anchor_cases))
    if block == 0:
        source_bank = "seen_anchor"
    else:
        if not 1 <= block <= len(selected):
            raise RuntimeError("case index is outside the transaction blocks")
        source_bank = f"fit_context_{int(selected[block - 1])}"
    return f"{source_bank}:{within:04d}"


def _split_transactions(transaction_rows, validation_fraction, salt):
    ranked = sorted(
        transaction_rows,
        key=lambda row: (
            _stable_digest(salt, row["transaction_id"]),
            row["transaction_index"],
        ),
    )
    if len(ranked) < 2:
        raise RuntimeError("multi-transaction split requires at least two rows")
    validation_count = max(
        1,
        min(
            len(ranked) - 1,
            int(round(len(ranked) * float(validation_fraction))),
        ),
    )
    validation = {
        row["transaction_id"] for row in ranked[:validation_count]
    }
    return {
        row["transaction_id"]: (
            "validation" if row["transaction_id"] in validation else "train"
        )
        for row in transaction_rows
    }


def _candidate_rows(artifact, transaction_indices):
    anchor_cases = int(artifact["anchor"]["bad"].shape[0])
    rows = []
    transaction_rows = []
    for transaction_index in transaction_indices:
        _, batch, selected = projected_probe._materialize_transaction(
            artifact, m.torch.device("cpu"), transaction_index
        )
        transaction_id = projected_probe._transaction_identity(
            transaction_index, selected
        )
        transaction_rows.append({
            "transaction_id": transaction_id,
            "transaction_index": int(transaction_index),
            "context_indices": list(selected),
        })
        for local_case_index in range(int(batch["bad"].shape[0])):
            group = m.REFINER_GROUP_LABELS[
                int(batch["group"][local_case_index].detach())
            ]
            source_case_uid = _source_case_identity(
                selected, anchor_cases, local_case_index
            )
            rows.append({
                "transaction_id": transaction_id,
                "transaction_index": int(transaction_index),
                "context_indices": list(selected),
                "case_index": int(local_case_index),
                "case_uid": f"{transaction_id}:{int(local_case_index)}",
                "source_case_uid": source_case_uid,
                "group": group,
            })
    return transaction_rows, rows


def build_manifest(
    artifact,
    *,
    fit_bank_sha256,
    source_diagnostic,
    max_transactions,
    max_cases_per_group,
    validation_fraction,
    split_salt=SPLIT_SALT,
):
    schedule = artifact.get("transaction_schedule") or []
    transaction_count = len(schedule)
    if transaction_count < 2:
        raise RuntimeError("fit bank does not contain multiple transactions")
    selected_count = min(int(max_transactions), transaction_count)
    transaction_indices = tuple(range(selected_count))
    transaction_rows, candidate_rows = _candidate_rows(
        artifact, transaction_indices
    )
    split_by_transaction = _split_transactions(
        transaction_rows, validation_fraction, split_salt
    )

    occurrences = defaultdict(lambda: {"train": [], "validation": []})
    for row in candidate_rows:
        split = split_by_transaction[row["transaction_id"]]
        occurrences[row["source_case_uid"]][split].append(row)

    selected = []
    for source_case_uid in sorted(occurrences):
        available = occurrences[source_case_uid]
        preferred = (
            "validation"
            if int(_stable_digest(split_salt, source_case_uid), 16) % 4 == 0
            else "train"
        )
        split = preferred if available[preferred] else (
            "train" if available["train"] else "validation"
        )
        row = min(
            available[split],
            key=lambda value: (
                value["transaction_index"], value["case_index"]
            ),
        )
        selected.append({**row, "split": split})

    by_transaction_group = defaultdict(list)
    for row in selected:
        role = "oracle" if str(row["group"]).startswith("cross_") else (
            "identity_control"
            if str(row["group"]).startswith("single_") else None
        )
        if role is None:
            continue
        row = {**row, "teacher_role": role}
        by_transaction_group[
            (row["transaction_id"], row["group"], role)
        ].append(row)

    retained = []
    for key in sorted(by_transaction_group):
        rows = sorted(
            by_transaction_group[key], key=lambda row: row["case_index"]
        )
        if key[2] == "oracle":
            rows = rows[:int(max_cases_per_group)]
        retained.extend(rows)

    retained_by_transaction = defaultdict(list)
    for row in retained:
        retained_by_transaction[row["transaction_id"]].append(row)

    transactions = []
    for transaction in transaction_rows:
        transaction_id = transaction["transaction_id"]
        cases = retained_by_transaction[transaction_id]
        split = split_by_transaction[transaction_id]
        transactions.append({
            **transaction,
            "split": split,
            "oracle_cases": [
                row for row in cases if row["teacher_role"] == "oracle"
            ],
            "identity_control_cases": [
                row
                for row in cases
                if row["teacher_role"] == "identity_control"
            ],
        })

    transactions = [
        row for row in transactions if row["oracle_cases"]
    ]
    retained_transaction_ids = {
        row["transaction_id"] for row in transactions
    }
    if len(transactions) < 2:
        raise RuntimeError(
            "fewer than two transactions retain independent Oracle cases"
        )
    if {row["split"] for row in transactions} != {"train", "validation"}:
        raise RuntimeError(
            "retained Oracle transactions do not cover both data splits"
        )
    retained = [
        row for row in retained
        if row["transaction_id"] in retained_transaction_ids
    ]

    train_uids = {
        row["case_uid"] for row in retained if row["split"] == "train"
    }
    validation_uids = {
        row["case_uid"]
        for row in retained if row["split"] == "validation"
    }
    train_sources = {
        row["source_case_uid"]
        for row in retained if row["split"] == "train"
    }
    validation_sources = {
        row["source_case_uid"]
        for row in retained if row["split"] == "validation"
    }
    if train_uids & validation_uids:
        raise RuntimeError("train/validation composite case keys overlap")
    if train_sources & validation_sources:
        raise RuntimeError("train/validation source cases overlap")

    counts = defaultdict(int)
    for row in retained:
        counts[(row["split"], row["group"], row["teacher_role"])] += 1
    payload = {
        "schema": SCHEMA,
        "development_only": True,
        "generated_before_oracle": True,
        "formal_training_allowed": False,
        "source_diagnostic": str(Path(source_diagnostic).resolve()),
        "fit_bank_sha256": fit_bank_sha256,
        "split_salt": split_salt,
        "split_algorithm": (
            "hashed_transaction_partition_plus_source_case_disjoint_owner_v1"
        ),
        "oracle_batch_domain": "complete_rotating_c5_transaction",
        "fixed_guard_anchor_policy": (
            "immutable_transaction_baseline_pre_oracle"
        ),
        "fixed_guard_tolerance_policy": (
            "unchanged_source_contract_relative_and_absolute_tolerances"
        ),
        "validation_fraction": float(validation_fraction),
        "available_transaction_count": transaction_count,
        "scheduled_transaction_count": selected_count,
        "selected_transaction_count": len(transactions),
        "max_cases_per_group_per_transaction": int(max_cases_per_group),
        "transactions": transactions,
        "train_case_uids": sorted(train_uids),
        "validation_case_uids": sorted(validation_uids),
        "train_source_case_uids": sorted(train_sources),
        "validation_source_case_uids": sorted(validation_sources),
        "train_validation_case_overlap": [],
        "train_validation_source_case_overlap": [],
        "counts": {
            f"{split}/{group}/{role}": count
            for (split, group, role), count in sorted(counts.items())
        },
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["manifest_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-diagnostic-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-transactions", type=int, default=8)
    parser.add_argument("--max-cases-per-group", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if args.max_transactions < 2:
        parser.error("--max-transactions must be at least two")
    if args.max_cases_per_group < 1:
        parser.error("--max-cases-per-group must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in (0, 1)")

    source = Path(args.source_diagnostic_dir)
    fit_bank_path = source / "fit_bank.pt"
    artifact = m.torch.load(
        fit_bank_path, map_location="cpu", weights_only=False
    )
    if artifact.get("formal_checkpoint") or not artifact.get("train_only"):
        raise RuntimeError("manifest source is not a development fit bank")
    payload = build_manifest(
        artifact,
        fit_bank_sha256=_file_sha256(fit_bank_path),
        source_diagnostic=source,
        max_transactions=args.max_transactions,
        max_cases_per_group=args.max_cases_per_group,
        validation_fraction=args.validation_fraction,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(
        f"{_file_sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({
        "stage": "v15_15e_transaction_manifest_complete",
        "manifest": str(output.resolve()),
        "manifest_file_sha256": _file_sha256(output),
        "manifest_content_sha256": payload["manifest_content_sha256"],
        "selected_transaction_count": payload["selected_transaction_count"],
        "counts": payload["counts"],
        "train_validation_case_overlap": [],
        "train_validation_source_case_overlap": [],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
