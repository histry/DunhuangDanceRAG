"""Freeze a result-blind deterministic case manifest from bank metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument(
        "--role",
        choices=("mechanism_train", "development", "sealed_held_out"),
        required=True,
    )
    parser.add_argument("--count", type=int)
    parser.add_argument(
        "--transaction-id",
        help=(
            "optional sealed transaction override; otherwise select the first "
            "eligible transaction by the frozen salted order"
        ),
    )
    parser.add_argument("--salt", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="previous train/development manifests whose transactions are forbidden",
    )
    args = parser.parse_args()
    bank_path = Path(args.bank).resolve()
    protocol_path = Path(args.protocol).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError("manifest output already exists")
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != "paper2_eval_protocol_v1":
        raise RuntimeError("paper2 protocol schema mismatch")
    if protocol.get("implementation_commit") != args.implementation_commit:
        raise RuntimeError("paper2 protocol is not frozen to the implementation")
    expected_split = "train" if args.role == "mechanism_train" else "validation"
    if bank.get("split") != expected_split:
        raise RuntimeError(
            f"{args.role} requires a {expected_split} bank"
        )
    if not bank.get("split_manifest_content_sha256"):
        raise RuntimeError("source bank lacks frozen split-manifest provenance")
    rows = []
    seen_case_uids = set()
    seen_source_case_uids = set()
    for sample in bank.get("samples", []):
        uid = str(sample["case_uid"])
        transaction = str(sample["transaction_id"])
        source_uid = str(sample.get("source_case_uid") or "")
        if not source_uid:
            raise RuntimeError(f"case {uid} lacks source_case_uid provenance")
        if uid in seen_case_uids:
            raise RuntimeError(f"source bank contains duplicate case UID: {uid}")
        if source_uid in seen_source_case_uids:
            raise RuntimeError(
                f"source bank reuses source_case_uid: {source_uid}"
            )
        seen_case_uids.add(uid)
        seen_source_case_uids.add(source_uid)
        score = hashlib.sha256(
            f"{args.salt}\0{transaction}\0{uid}".encode("utf-8")
        ).hexdigest()
        rows.append({
            "score": score,
            "transaction_id": transaction,
            "case_uid": uid,
            "source_case_uid": source_uid,
        })
    if not rows:
        raise RuntimeError("bank contains no samples")
    rows.sort(key=lambda row: (
        row["score"], row["transaction_id"], row["case_uid"]
    ))
    excluded_transactions = set()
    excluded_sources = set()
    for excluded_path in args.exclude_manifest:
        excluded = json.loads(
            Path(excluded_path).read_text(encoding="utf-8")
        )
        if excluded.get("schema") != "paper2_case_manifest_v1":
            raise RuntimeError("excluded paper2 manifest schema mismatch")
        excluded_content = dict(excluded)
        excluded_sha256 = excluded_content.pop("content_sha256", None)
        if excluded_sha256 != _canonical_sha(excluded_content):
            raise RuntimeError("excluded paper2 manifest content SHA256 mismatch")
        if excluded.get("implementation_commit") != args.implementation_commit:
            raise RuntimeError("excluded manifest implementation commit mismatch")
        if excluded.get("protocol_sha256") != _file_sha(protocol_path):
            raise RuntimeError("excluded manifest protocol SHA256 mismatch")
        excluded_transactions.update(
            str(value) for value in excluded.get("transaction_ids", [])
        )
        excluded_sources.update(
            str(value) for value in excluded.get("source_case_uids", [])
        )
    rows_by_transaction = {}
    for row in rows:
        rows_by_transaction.setdefault(row["transaction_id"], []).append(row)
    eligible_transactions = {
        transaction_id
        for transaction_id, transaction_rows in rows_by_transaction.items()
        if transaction_id not in excluded_transactions
        and not {
            row["source_case_uid"] for row in transaction_rows
        } & excluded_sources
    }
    eligible = [
        row for row in rows
        if row["transaction_id"] in eligible_transactions
    ]
    if args.role == "sealed_held_out":
        if args.count is not None:
            raise RuntimeError("sealed manifest forbids --count")
        transactions = sorted({
            row["transaction_id"] for row in eligible
        }, key=lambda value: (
            hashlib.sha256(
                f"{args.salt}\0{value}".encode("utf-8")
            ).hexdigest(),
            value,
        ))
        selected_transaction = args.transaction_id or (
            transactions[0] if transactions else None
        )
        selected = [
            row for row in eligible
            if row["transaction_id"] == selected_transaction
        ]
        if not selected:
            raise RuntimeError(
                "sealed transaction is absent, reused, or source-overlapping"
            )
    elif args.transaction_id:
        raise RuntimeError("--transaction-id is sealed-only")
    else:
        count = int(args.count or len(eligible))
        if args.role == "mechanism_train" and not 12 <= count <= 20:
            raise RuntimeError("mechanism manifest count must be in [12, 20]")
        if not 1 <= count <= len(eligible):
            raise RuntimeError("manifest count is outside the eligible bank")
        selected = eligible[:count]
    payload = {
        "schema": "paper2_case_manifest_v1",
        "immutable": True,
        "role": args.role,
        "case_uids": [row["case_uid"] for row in selected],
        "transaction_ids": sorted({
            row["transaction_id"] for row in selected
        }),
        "source_case_uids": sorted({
            row["source_case_uid"] for row in selected
        }),
        "selection_rule": (
            "all_cases_from_seeded_unseen_transaction"
            if args.role == "sealed_held_out"
            else "sha256(salt,transaction_id,case_uid)_ascending"
        ),
        "selection_salt": args.salt,
        "result_fields_consumed": False,
        "created_before_results": True,
        "source_bank": str(bank_path),
        "source_bank_sha256": _file_sha(bank_path),
        "implementation_commit": args.implementation_commit,
        "protocol": str(protocol_path),
        "protocol_sha256": _file_sha(protocol_path),
        "source_split_manifest_content_sha256": bank.get(
            "split_manifest_content_sha256"
        ),
        "excluded_transaction_ids": sorted(excluded_transactions),
        "excluded_source_case_uids": sorted(excluded_sources),
    }
    payload["content_sha256"] = _canonical_sha(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": "paper2_case_manifest_frozen",
        "output": str(output),
        "case_count": len(selected),
    }), flush=True)


if __name__ == "__main__":
    main()
