import inspect
import json
from pathlib import Path

import pytest
import torch

from training import refiner_rcsp_single_direction_attribution as a
from training import refiner_role_conditioned_support_projection_experiment as rcsp
from tests.test_refiner_role_conditioned_support_projection import nonzero_base, tiny_batch


def test_schema_source_and_fixed_transaction_contract():
    assert a.SCHEMA == "refiner_rcsp_single_direction_attribution_v1"
    assert a.RCSP_SOURCE_COMMIT == "5a344f2950183ceb4c8e938a3c26fa5d76a78c3f"
    assert a.TRANSACTION_INDEX == 0
    assert a.GROUP_SPEC == {
        "single_short": ("single_recording", 10, 0),
        "single_long": ("single_recording", 28, 1),
        "cross_short": ("cross_event", 10, 2),
        "cross_long": ("cross_event", 28, 3),
    }


def test_final_split_uses_explicit_metadata_for_balanced_role_width_groups():
    batch = {"clean": torch.arange(64).reshape(64, 1)}
    metadata = []
    for split, role in rcsp.FINAL_BLOCK_ORDER:
        for width in (10, 28):
            metadata.extend({"split": split, "role": role, "width": width} for _ in range(8))
    seen = a.final_split_batch(batch, metadata, "seen")
    assert seen["clean"].shape[0] == 32
    assert torch.bincount(seen["group"], minlength=4).tolist() == [8, 8, 8, 8]
    assert seen["role_id"].tolist() == [0] * 16 + [1] * 16
    metadata[0]["width"] = 99
    with pytest.raises(ValueError, match="does not map"):
        a.final_split_batch(batch, metadata, "seen")


def test_parameter_gradient_geometry_is_read_only_and_role_local():
    cfg, batch = tiny_batch(cases=4)
    model = rcsp.FrozenBaseRCSPModel(nonzero_base())
    with torch.no_grad():
        model.adapter.single_adapter.weight.normal_(0, 1.0e-4)
        model.adapter.cross_adapter.weight.normal_(0, 1.0e-4)
    batch = rcsp.attach_train_role_ids(batch)
    base_hash = a.safe.state_hash(model.base.state_dict())
    adapter_hash = a.safe.state_hash(model.adapter.state_dict())
    rows, vectors = a.parameter_gradient_geometry(model, batch, cfg, "train_transaction_0")
    assert len(rows) == 4
    assert set(vectors) == {
        "train_transaction_0/single_recording/10",
        "train_transaction_0/single_recording/28",
        "train_transaction_0/cross_event/10",
        "train_transaction_0/cross_event/28",
    }
    assert all(row["cases"] == 1 for row in rows)
    assert all(row["optimizer_update_performed"] is False for row in rows)
    assert a.safe.state_hash(model.base.state_dict()) == base_hash
    assert a.safe.state_hash(model.adapter.state_dict()) == adapter_hash
    assert all(parameter.grad is None for parameter in model.parameters())


def geometry_fixture():
    rows, vectors = [], {}
    for role in rcsp.ROLE_MAPPING:
        for source_index, source in enumerate(a.SOURCE_ORDER):
            for width_index, width in enumerate((10, 28)):
                key = f"{source}/{role}/{width}"
                sign = -1.0 if role == "single_recording" and source_index == 0 and width == 28 else 1.0
                vector = torch.tensor([sign, float(source_index + 1), float(width_index)])
                vectors[key] = vector
                rows.append(
                    {
                        "key": key,
                        "source": source,
                        "role": role,
                        "width": width,
                        "parameter_gradient_norm": float(vector.norm()),
                        "learned_displacement_vs_negative_gradient_cosine": 0.2,
                    }
                )
    return rows, vectors


def test_gradient_tables_keep_roles_separate_and_report_signed_conflicts():
    rows, vectors = geometry_fixture()
    # Make only the TRAIN single width gradients exactly opposite.
    vectors["train_transaction_0/single_recording/10"] = torch.tensor([1.0, 0.0, 0.0])
    vectors["train_transaction_0/single_recording/28"] = torch.tensor([-1.0, 0.0, 0.0])
    tables = a.gradient_cosine_tables(rows, vectors)
    assert tables["single_recording"]["same_source_width_10_vs_28"][
        "train_transaction_0"
    ] == pytest.approx(-1.0)
    assert tables["single_recording"]["negative_cosine_pairs"]
    assert tables["cross_event"]["order"][0].endswith("cross_event/10")


def test_scientific_answer_uses_sign_conflict_without_claiming_root_cause():
    rows, vectors = geometry_fixture()
    vectors["train_transaction_0/single_recording/10"] = torch.tensor([1.0, 0.0, 0.0])
    vectors["train_transaction_0/single_recording/28"] = torch.tensor([-1.0, 0.0, 0.0])
    tables = a.gradient_cosine_tables(rows, vectors)
    existing = {
        "descriptive_ratios": {"single_to_cross_projected_direction_cosine_ratio": 0.01},
        "support_projection_summary": {
            "width:10": {"projection_retention_ratio_median": 0.5},
            "width:28": {"projection_retention_ratio_median": 0.6},
        },
    }
    answer = a.scientific_answers(rows, tables, existing)
    assert answer["single_direction_attribution"] == (
        "SINGLE_HEAD_WITHIN_ROLE_WIDTH_GRADIENT_CONFLICT_OBSERVED"
    )
    assert answer["single_same_source_width_gradient_conflict"]["train_transaction_0"]
    assert answer["hard_support_escape_observed"] is False
    assert "does not prove" in answer["claim_boundary"]


def test_completed_rcsp_artifact_loader_checks_hashes_review_and_false_flags(tmp_path):
    checkpoint_path = tmp_path / "adapter_final.pt"
    torch.save(
        {
            "schema": a.RCSP_SOURCE_SCHEMA,
            "completed_steps": rcsp.STEPS,
            "adapter_state_dict": {},
            "formal_checkpoint": False,
            "production_model_modified": False,
            "checkpoint_selection_performed": False,
            "publish_allowed": False,
            "pilot_allowed": False,
            "resume_allowed": False,
        },
        checkpoint_path,
    )
    report = {
        "schema": a.RCSP_SOURCE_SCHEMA,
        "completed": True,
        "provenance": {"runtime_commit": a.RCSP_SOURCE_COMMIT},
        "parameter_update_scope": {
            "adapter_checkpoint": {"sha256": a._file_sha256(checkpoint_path)}
        },
        "checkpoint_selection_performed": False,
        "scale_selection_performed": False,
        "production_model_modified": False,
        "production_inference_modified": False,
        "scientific_acceptance": False,
        "publish_allowed": False,
        "pilot_allowed": False,
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    review = {
        "schema": a.RCSP_REVIEW_SCHEMA,
        "completed": True,
        "source_report": {"sha256": a._file_sha256(report_path)},
        "measurement_recomputation_verified": True,
        "formal_conclusion": {
            "classification": (
                "ROLE_CONDITIONING_USEFUL_BUT_WIDTH_DEPENDENT_MECHANISM_REMAINS"
            ),
            "role_conditioning_alone_sufficient": False,
        },
        "production_model_modified": False,
        "scientific_acceptance": False,
        "pilot_allowed": False,
    }
    (tmp_path / "reporting_logic_review_v1.json").write_text(
        json.dumps(review), encoding="utf-8"
    )
    loaded = a._load_rcsp_artifacts(tmp_path, a.RCSP_SOURCE_COMMIT)
    assert loaded[3]["completed"] is True
    assert loaded[4]["measurement_recomputation_verified"] is True

    report["pilot_allowed"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostic-only RCSP"):
        a._load_rcsp_artifacts(tmp_path, a.RCSP_SOURCE_COMMIT)


def test_cli_and_source_have_no_training_or_selection_controls():
    cli = inspect.getsource(a.main)
    for forbidden in ("--steps", "--alpha", "--width", "--seed", "--resume"):
        assert forbidden not in cli
    source = Path(a.__file__).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert '"optimizer_steps": 0' in source
    assert '"parameter_update_performed": False' in source
    assert '"production_model_modified": False' in source
    assert '"pilot_allowed": False' in source


def test_shell_requires_clean_exact_main_and_forbids_optimizer_and_pilot():
    source = (
        Path(__file__).parents[1] / "scripts" / "audit_refiner_rcsp_single_direction.sh"
    ).read_text(encoding="utf-8")
    assert "git status --porcelain" in source
    assert "git rev-parse origin/main" in source
    assert "reporting_logic_review_v1.json" in source
    assert "No optimizer" in source
    assert "Pilot remains forbidden" in source
