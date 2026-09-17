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
        help="required for sealed role; include every case from this new transaction",
    )
    parser.add_argument("--salt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="previous train/development manifests whose transactions are forbidden",
    )
    args = parser.parse_args()
    bank_path = Path(args.bank).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError("manifest output already exists")
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    expected_split = "train" if args.role == "mechanism_train" else "validation"
    if bank.get("split") != expected_split:
        raise RuntimeError(
            f"{args.role} requires a {expected_split} bank"
        )
    rows = []
    for sample in bank.get("samples", []):
        uid = str(sample["case_uid"])
        transaction = str(sample["transaction_id"])
        score = hashlib.sha256(
            f"{args.salt}\0{transaction}\0{uid}".encode("utf-8")
        ).hexdigest()
        rows.append((score, transaction, uid))
    if not rows:
        raise RuntimeError("bank contains no samples")
    rows.sort()
    excluded_transactions = set()
    for excluded_path in args.exclude_manifest:
        excluded = json.loads(
            Path(excluded_path).read_text(encoding="utf-8")
        )
        excluded_transactions.update(
            str(value) for value in excluded.get("transaction_ids", [])
        )
    if args.role == "sealed_held_out":
        if not args.transaction_id or args.count is not None:
            raise RuntimeError(
                "sealed manifest requires --transaction-id and forbids --count"
            )
        rows = [row for row in rows if row[1] == args.transaction_id]
        if not rows:
            raise RuntimeError("sealed transaction is absent from the bank")
        if args.transaction_id in excluded_transactions:
            raise RuntimeError("sealed transaction overlaps prior evidence")
    elif args.transaction_id:
        raise RuntimeError("--transaction-id is sealed-only")
    count = int(args.count or len(rows))
    if args.role == "mechanism_train" and not 12 <= count <= 20:
        raise RuntimeError("mechanism manifest count must be in [12, 20]")
    if not 1 <= count <= len(rows):
        raise RuntimeError("manifest count is outside the bank")
    selected = rows[:count]
    payload = {
        "schema": "paper2_case_manifest_v1",
        "role": args.role,
        "case_uids": [uid for _, _, uid in selected],
        "transaction_ids": sorted({txn for _, txn, _ in selected}),
        "selection_rule": (
            "all_cases_from_predeclared_new_transaction"
            if args.role == "sealed_held_out"
            else "sha256(salt,transaction_id,case_uid)_ascending"
        ),
        "selection_salt": args.salt,
        "result_fields_consumed": False,
        "created_before_results": True,
        "source_bank": str(bank_path),
        "source_bank_sha256": _file_sha(bank_path),
        "source_split_manifest_content_sha256": bank.get(
            "split_manifest_content_sha256"
        ),
        "excluded_transaction_ids": sorted(excluded_transactions),
    }
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
