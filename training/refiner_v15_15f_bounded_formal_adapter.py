"""V15.15f bounded formal Adapter training with split-safe closure audits.

This runner trains only on the frozen V15.15e train bank.  It resumes both
Adapter and AdamW state between bounded chunks, then audits the train and
validation banks independently after every chunk.  A failed exact closure is
reported fail-closed and never produces a promotable checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from training import motion_models as m


SCHEMA = "refiner_v15_15f_bounded_formal_adapter_training_v1"
TEACHER_SCHEMA = (
    "refiner_v15_15e_multi_transaction_observable_adapter_teacher_bank_v1"
)
FIXED_FLAGS = (
    "fixed_guard_thresholds_changed",
    "physical_gate_changed",
    "fixed_support_gate_changed",
    "fidelity_gate_changed",
    "boundary_gate_changed",
    "observable_0p03_gate_changed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bank(path: Path, expected_split: str) -> dict:
    payload = m.torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != TEACHER_SCHEMA:
        raise RuntimeError(f"unsupported teacher bank: {path}")
    if payload.get("split") != expected_split:
        raise RuntimeError(
            f"teacher bank split mismatch: expected {expected_split}"
        )
    if not payload.get("teacher_bank_ready"):
        raise RuntimeError(f"{expected_split} teacher bank is not ready")
    changed = [key for key in FIXED_FLAGS if payload.get(key) is not False]
    if changed:
        raise RuntimeError({"teacher_bank_changed_fixed_contract": changed})
    evidence = payload.get("samples", [])
    case_uids = [str(row["case_uid"]) for row in evidence]
    if len(case_uids) != len(set(case_uids)):
        raise RuntimeError("teacher bank contains duplicate composite keys")
    return payload


def _split_contract(train: dict, validation: dict) -> dict:
    if train.get("split_manifest_content_sha256") != validation.get(
        "split_manifest_content_sha256"
    ):
        raise RuntimeError("train and validation manifest hashes differ")
    train_uids = {str(row["case_uid"]) for row in train["samples"]}
    validation_uids = {
        str(row["case_uid"]) for row in validation["samples"]
    }
    train_sources = {str(row["source_case_uid"]) for row in train["samples"]}
    validation_sources = {
        str(row["source_case_uid"]) for row in validation["samples"]
    }
    case_overlap = sorted(train_uids & validation_uids)
    source_overlap = sorted(train_sources & validation_sources)
    if case_overlap or source_overlap:
        raise RuntimeError({
            "train_validation_case_overlap": case_overlap,
            "train_validation_source_case_overlap": source_overlap,
        })
    return {
        "manifest_content_sha256": train["split_manifest_content_sha256"],
        "train_validation_case_overlap": case_overlap,
        "train_validation_source_case_overlap": source_overlap,
        "validation_gate_floor_consumed_for_calibration": False,
    }


def _probe_command(
    args,
    *,
    teacher_bank: Path,
    adapter_state: Path,
    output_dir: Path,
    steps: int,
    audit_only: bool,
    resume_optimizer: bool,
    preserve_gate_floor: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "training.refiner_observable_adapter_probe",
        "--config",
        args.config,
        "--teacher-bank",
        str(teacher_bank),
        "--adapter-state",
        str(adapter_state),
        "--exact-radius-training",
        "--case-isolated-guard-restoration",
        "--output-dir",
        str(output_dir),
        "--steps",
        str(int(steps)),
        "--eval-every",
        str(max(1, int(steps))),
        "--learning-rate",
        str(float(args.learning_rate)),
        "--gradient-clip",
        str(float(args.gradient_clip)),
        "--target-rms",
        "1e-4",
        "--normalization-eps",
        "1e-8",
        "--nonregression-weight",
        "25",
        "--case20-temporal-weight",
        "25",
        "--guard-safety-fraction",
        "0.25",
        "--guard-restoration-weight",
        "1",
        "--guard-direction-floor",
        "0.1",
        "--guard-direction-decay",
        "1",
    ]
    if audit_only:
        command.append("--audit-only")
    if resume_optimizer:
        command.append("--resume-optimizer")
    if preserve_gate_floor:
        command.append("--preserve-adapter-gate-floor")
    return command


def _run_probe(command: list[str], output_dir: Path) -> tuple[int, dict]:
    output_dir.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run(command, check=False)
    report_path = output_dir / "observable_adapter_probe.report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8-sig"))
        if report_path.is_file() else {}
    )
    return int(completed.returncode), report


def _audit_passed(report: dict) -> bool:
    criteria = report.get("formal_readiness_criteria", {})
    return bool(
        report.get("ready_for_formal_adapter_training")
        and report.get("scope_safe")
        and report.get("numeric_audit_complete")
        and criteria
        and all(bool(value) for value in criteria.values())
    )


def run(args) -> int:
    started = time.perf_counter()
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    train_path = Path(args.train_teacher_bank).resolve()
    validation_path = Path(args.validation_teacher_bank).resolve()
    initial_state = Path(args.adapter_state).resolve()
    for path in (train_path, validation_path, initial_state):
        if not path.is_file():
            raise FileNotFoundError(path)

    train = _load_bank(train_path, "train")
    validation = _load_bank(validation_path, "validation")
    split_contract = _split_contract(train, validation)
    frozen_inputs = {
        "train_teacher_bank": _sha256(train_path),
        "validation_teacher_bank": _sha256(validation_path),
        "initial_adapter_state": _sha256(initial_state),
    }
    (destination / "frozen_inputs.sha256.json").write_text(
        json.dumps(frozen_inputs, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rounds = []
    current_state = initial_state
    completed_steps = 0
    fail_reason = None
    while completed_steps < int(args.total_steps):
        chunk = min(
            int(args.audit_every), int(args.total_steps) - completed_steps
        )
        step_start = completed_steps
        step_stop = completed_steps + chunk
        round_index = len(rounds) + 1
        round_root = destination / f"round_{round_index:03d}"
        train_dir = round_root / "train"
        train_command = _probe_command(
            args,
            teacher_bank=train_path,
            adapter_state=current_state,
            output_dir=train_dir,
            steps=chunk,
            audit_only=False,
            resume_optimizer=bool(round_index > 1),
            preserve_gate_floor=bool(round_index > 1),
        )
        train_status, train_report = _run_probe(train_command, train_dir)
        next_state = train_dir / "observable_adapter_probe_state.pt"
        train_passed = bool(
            train_status == 0 and next_state.is_file()
            and _audit_passed(train_report)
        )
        round_record = {
            "round": round_index,
            "step_start": step_start,
            "step_stop": step_stop,
            "train_status": train_status,
            "train_report": str(
                train_dir / "observable_adapter_probe.report.json"
            ),
            "train_passed": train_passed,
            "optimizer_state_resumed": bool(round_index > 1),
        }
        completed_steps = step_stop
        if not train_passed:
            fail_reason = "train_exact_closure_failed"
            rounds.append(round_record)
            break

        validation_dir = round_root / "validation"
        validation_command = _probe_command(
            args,
            teacher_bank=validation_path,
            adapter_state=next_state,
            output_dir=validation_dir,
            steps=0,
            audit_only=True,
            resume_optimizer=False,
            preserve_gate_floor=True,
        )
        validation_status, validation_report = _run_probe(
            validation_command, validation_dir
        )
        validation_passed = bool(
            validation_status == 0 and _audit_passed(validation_report)
        )
        round_record.update({
            "validation_status": validation_status,
            "validation_report": str(
                validation_dir / "observable_adapter_probe.report.json"
            ),
            "validation_passed": validation_passed,
            "validation_gate_floor_source": validation_report.get(
                "adapter_gate_floor_source"
            ),
        })
        rounds.append(round_record)
        current_state = next_state
        print(json.dumps({
            "stage": "v15_15f_bounded_training_round",
            **round_record,
        }), flush=True)
        if not validation_passed:
            fail_reason = "validation_exact_closure_failed"
            break

    passed = bool(
        fail_reason is None and completed_steps == int(args.total_steps)
    )
    checkpoint_path = None
    if passed:
        payload = m.torch.load(
            current_state, map_location="cpu", weights_only=False
        )
        payload.update({
            "schema": SCHEMA,
            "formal_checkpoint": True,
            "formal_training_trial": True,
            "promotion_allowed": False,
            "publish_allowed": False,
            "total_steps": int(completed_steps),
            "train_teacher_bank_sha256": frozen_inputs[
                "train_teacher_bank"
            ],
            "validation_teacher_bank_sha256": frozen_inputs[
                "validation_teacher_bank"
            ],
        })
        checkpoint_path = destination / "bounded_formal_adapter_state.pt"
        m.torch.save(payload, checkpoint_path)

    report = {
        "schema": SCHEMA,
        "implementation_commit": os.environ.get("EXPECTED_COMMIT"),
        "bounded_formal_training": True,
        "promotion_allowed": False,
        "publish_allowed": False,
        "pseudo_teachers_generated": False,
        "new_optimization_layer_enabled": False,
        "target_rms": 1.0e-4,
        "total_steps_requested": int(args.total_steps),
        "total_steps_completed": int(completed_steps),
        "audit_every": int(args.audit_every),
        "optimizer_state_continuity": "adamw_state_resumed_between_chunks",
        "train_teacher_bank": str(train_path),
        "validation_teacher_bank": str(validation_path),
        "frozen_input_sha256": frozen_inputs,
        "split_contract": split_contract,
        "rounds": rounds,
        "fail_closed_reason": fail_reason,
        "bounded_training_passed": passed,
        "formal_checkpoint": (
            str(checkpoint_path) if checkpoint_path is not None else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = destination / "bounded_formal_adapter.report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": "v15_15f_bounded_formal_adapter_complete",
        "passed": passed,
        "report": str(report_path),
        "checkpoint": report["formal_checkpoint"],
        "fail_closed_reason": fail_reason,
    }), flush=True)
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-teacher-bank", required=True)
    parser.add_argument("--validation-teacher-bank", required=True)
    parser.add_argument("--adapter-state", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/motion_model.json")
    parser.add_argument("--total-steps", type=int, default=300)
    parser.add_argument("--audit-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    args = parser.parse_args()
    if not 200 <= args.total_steps <= 500:
        parser.error("--total-steps must remain in the bounded [200, 500] range")
    if not 10 <= args.audit_every <= 20:
        parser.error("--audit-every must remain in [10, 20]")
    if args.total_steps % args.audit_every:
        parser.error("--total-steps must be divisible by --audit-every")
    if args.learning_rate <= 0.0 or args.gradient_clip <= 0.0:
        parser.error("learning rate and gradient clip must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
