import inspect
import json

import pytest
import torch

from training import motion_models as m
from training import refiner_final_failure_audit as failure
from training import refiner_temporal_action_alignment_audit as alignment
from training import refiner_temporal_scale_response_audit as s
from tests.test_refiner_group_gradient_audit import bank_tensor
from tests.test_refiner_temporal_action_alignment_audit import small_batch


def trace_tensor(cases=1, frames=2, value=1.0):
    raw = torch.full((cases, frames, 75), value)
    trace = {name: raw.clone() for name in s.TRACE_STAGES}
    trace.update(root_mask=torch.ones(cases, frames, 1),
                 joint_mask=torch.ones(cases, frames, 24),
                 root_cap_m=.08, rotation_cap_rad=.35)
    return trace


def alignment_payload(traj_hashes, state_hash, source_hashes, probe_hash):
    answers = {
        "train_temporal_gradient_present": {"answer": True},
        "final_temporal_gradient_present": {"answer": True},
        "final_model_action_vs_zero_origin_temporal_descent": {
            "answer": "mostly_aligned_with_negative_gradient"},
        "current_temporal_action_scaling": {
            "answer": "increasing_current_action_scale_is_mostly_local_descent"},
        "endpoint_control_at_zero_origin": {"answer": "mostly_aligned_with_negative_gradient"},
        "gradient_outside_decoder_support": {"all_exact_zero": True, "nonzero_measurements": 0},
    }
    return {"schema": alignment.SCHEMA, "optimizer_steps": 0,
            "parameter_update_performed": False, "model_state_unchanged": True,
            "scientific_acceptance": False, "publish_allowed": False, "pilot_allowed": False,
            "provenance": {"runtime_commit": s.REVIEWED_ALIGNMENT_COMMIT,
                           "trajectory_sha256": traj_hashes,
                           "trajectory_final_state_sha256": state_hash,
                           "source_sha256_including_probe": source_hashes,
                           "probe_sha256": probe_hash},
            "scientific_answers": answers}


def curve_point(alpha, objective, derivative, passes=0):
    return {"alpha": alpha, "temporal_scientific_deficit_mean": objective,
            "dL_temporal_dalpha": derivative, "temporal_gate_pass_cases": passes}


def parity_fixture():
    trajectory = {"metrics": {}}
    actual = []
    for split in ("seen", "new_position"):
        single, cross = [], []
        for role, target in (("single_recording", single), ("cross_event", cross)):
            for index in range(16):
                width = 10 if index % 2 == 0 else 28
                observable = {"after": {"temporal_energy": 2.0,
                                         "endpoint_velocity_jump_mps": 3.0},
                              "temporal_gain": .01, "endpoint_gain": .02,
                              "temporal_accepted": False, "endpoint_accepted": False,
                              "reference_fidelity_accepted": True,
                              "reference_fidelity": {"fk_p95_m": 0.0, "fk_max_m": 0.0,
                                                     "product_log_l1": 0.0}}
                expected = {"case_index": index, "width": width, "observable": observable}
                if role == "single_recording":
                    observable["physical_non_regression"] = {"accepted": True}
                    expected["clean_identity"] = {"accepted": True}
                else:
                    expected["safety"] = {"accepted": True}
                target.append(expected)
                actual.append({"split": split, "role": role, "width": width,
                    "bank_case_index": index,
                    "responses": {"1.00": {
                        "authoritative_observable": {"temporal_metric": 2.0, "endpoint_metric": 3.0,
                            "temporal_repair_gain": .01, "endpoint_repair_gain": .02,
                            "temporal_gate_pass": False, "endpoint_gate_pass": False},
                        "geometry": {"accepted": True, "reference_fidelity": {
                            "fk_p95_m": 0.0, "fk_max_m": 0.0, "product_log_l1": 0.0}},
                        "physical": {"accepted": True, "reasons": [], "authoritative_gate": {}},
                        "clean_identity": {"accepted": True, "product_log_l1": 0.0,
                                           "contact_l1": 0.0}}}})
        trajectory["metrics"][split] = {"windows": single, "cross_event": {"windows": cross}}
    return actual, trajectory


def test_schema_and_reviewed_alignment_commit_are_fixed():
    assert s.SCHEMA == "refiner_temporal_scale_response_audit_v1"
    assert s.REVIEWED_ALIGNMENT_COMMIT == "5557f78398f94e448c61e8d14bbf25ac0d5ee373"


def test_alpha_grid_is_exact_and_ordered():
    assert s.ALPHAS == (0.0, .5, .75, 1.0, 1.25, 1.5, 2.0)
    assert s.ALPHA_KEYS == ("0.00", "0.50", "0.75", "1.00", "1.25", "1.50", "2.00")


@pytest.mark.parametrize("alpha", s.ALPHAS)
def test_scaling_preserves_contact_and_scales_only_geometry(alpha):
    raw = torch.randn(2, 3, 79)
    scaled = s.scale_raw_output(raw, alpha)
    assert torch.equal(scaled[..., :4], raw[..., :4])
    torch.testing.assert_close(scaled[..., 4:], alpha * raw[..., 4:], rtol=0, atol=0)


def test_alpha_zero_geometry_is_exact_zero_even_with_nonzero_contact():
    raw = torch.ones(1, 2, 79)
    scaled = s.scale_raw_output(raw, 0.0)
    assert bool((scaled[..., 4:] == 0).all()) and bool((scaled[..., :4] == 1).all())


def test_alpha_one_raw_output_is_exact_identity():
    raw = torch.randn(1, 2, 79)
    assert torch.equal(s.scale_raw_output(raw, 1.0), raw)


@pytest.mark.parametrize("failure", ["shape", "nonfinite", "nonscalar"])
def test_invalid_scale_inputs_fail_closed(failure):
    raw = torch.zeros(1, 2, 79 if failure != "shape" else 78)
    alpha = float("nan") if failure == "nonfinite" else torch.ones(2) if failure == "nonscalar" else 1.0
    with pytest.raises(ValueError):
        s.scale_raw_output(raw, alpha)


def test_attenuation_ratio_has_null_zero_denominator():
    assert s.ratio(1.0, 0.0) is None
    assert s.ratio(1.0, 2.0) == .5


def test_tensor_stats_report_l2_rms_and_max():
    stats = s.tensor_stats(torch.tensor([3.0, 4.0]))
    assert stats["l2_norm"] == 5 and stats["rms"] == pytest.approx(5 / 2**.5)
    assert stats["abs_max"] == 4


def test_decoder_trace_uses_actual_six_stage_order_and_ratios():
    trace = trace_tensor(value=2.0)
    trace["after_mask"].mul_(.5)
    trace["applied"].mul_(.25)
    rows = s.decoder_case_stats(trace, torch.ones(1, 2, 75))
    assert rows[0]["stage_order"] == list(s.TRACE_STAGES)
    assert rows[0]["attenuation"]["mask_attenuation_l2_ratio"] == pytest.approx(.5)
    assert rows[0]["attenuation"]["total_attenuation_l2_ratio"] == pytest.approx(.25)


def test_decoder_stats_reject_missing_production_stage():
    trace = trace_tensor()
    del trace["after_cap"]
    with pytest.raises(ValueError, match="incomplete"):
        s.decoder_case_stats(trace, torch.ones(1, 2, 75))


def test_root_cap_saturation_uses_real_vector_norm():
    trace = trace_tensor(value=0.0)
    trace["after_taper"][..., 0] = .081
    row = s.cap_saturation_stats(trace)[0]
    assert row["root_cap_saturation_fraction"] == 1
    assert row["block_saturation_fraction"]["root"] == 1


def test_body_cap_saturation_uses_canonical_complement():
    trace = trace_tensor(value=0.0)
    trace["after_taper"][..., 3:6] = .3
    row = s.cap_saturation_stats(trace)[0]
    assert row["block_saturation_fraction"]["body"] > 0
    assert row["block_saturation_fraction"]["extremity"] == 0


def test_extremity_cap_saturation_uses_canonical_indices():
    trace = trace_tensor(value=0.0)
    joint = alignment.EXTREMITY_JOINTS[0]
    trace["after_taper"][..., 3 + 3 * joint:6 + 3 * joint] = .3
    row = s.cap_saturation_stats(trace)[0]
    assert row["block_saturation_fraction"]["extremity"] > 0
    assert row["block_saturation_fraction"]["body"] == 0


def test_no_cap_saturation_reports_zero_frames_and_cases():
    row = s.cap_saturation_stats(trace_tensor(value=0.0))[0]
    assert row["frames_with_any_saturation"] == 0
    assert not row["case_has_any_saturation"]


@pytest.mark.parametrize("alpha", s.ALPHAS)
@pytest.mark.parametrize("coefficient", [2.0, 3.0], ids=["temporal", "endpoint"])
def test_fixed_h_finite_difference_matches_autograd_at_every_alpha(alpha, coefficient):
    result = s.synthetic_finite_difference(lambda value: coefficient * value.square(), alpha)
    assert result["h"] == s.FD_H
    assert result["autograd"] == pytest.approx(result["finite_difference"], abs=1e-9)


@pytest.mark.parametrize("failure", ["h", "alpha"])
def test_finite_difference_contract_cannot_adapt(failure):
    kwargs = {"h": .01} if failure == "h" else {}
    alpha = .33 if failure == "alpha" else 1.0
    with pytest.raises(ValueError, match="fixed"):
        s.synthetic_finite_difference(lambda value: value.square(), alpha, **kwargs)


def test_pearson_and_spearman_are_descriptive_only():
    result = s.correlation(list(range(7)), list(range(7)))
    assert result["pearson"] == pytest.approx(1) and result["spearman"] == pytest.approx(1)
    assert result["descriptive_only"] and not result["statistical_significance_claim"]


def test_constant_response_has_null_correlation():
    result = s.correlation([1] * 7, list(range(7)))
    assert result["pearson"] is None and result["spearman"] is None


def test_monotonic_curve_classification():
    points = [curve_point(alpha, 7 - index, -1) for index, alpha in enumerate(s.ALPHAS)]
    shape = s._curve_shape(points)
    assert shape["monotonic_improvement_over_grid"] and not shape["local_turning_detected"]


def test_turning_curve_classification_uses_fixed_grid_derivative_signs():
    points = [curve_point(alpha, float((alpha - .75) ** 2), alpha - .75) for alpha in s.ALPHAS]
    shape = s._curve_shape(points)
    assert shape["local_turning_detected"] and not shape["monotonic_improvement_over_grid"]


def test_gate_crossing_lists_all_points_without_single_point_selection():
    points = [curve_point(alpha, 1.0, -1, passes=int(alpha >= 1.25)) for alpha in s.ALPHAS]
    assert s._curve_shape(points)["temporal_gate_crossing_alphas"] == [1.25, 1.5, 2.0]


def test_train_layout_requires_192_cases_and_48_per_group():
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    part = bank_tensor(cfg)
    batch = {key: torch.cat([value] * 6) for key, value in part.items()}
    metadata = alignment.train_metadata(batch)
    assert len(metadata) == 192
    assert {group: sum(row["group"] == group for row in metadata)
            for group in m.REFINER_GROUP_LABELS} == {group: 48 for group in m.REFINER_GROUP_LABELS}


def test_fixed_final_layout_has_64_and_eight_cells():
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    part = {key: value[:16] for key, value in bank_tensor(cfg).items() if key != "group"}
    banks = {(split, role): part for split in ("seen", "new_position")
             for role in ("single_recording", "cross_event")}
    batch, metadata = alignment.combine_final_banks(banks)
    assert batch["clean"].shape[0] == len(metadata) == 64
    assert len({(row["split"], row["role"], row["width"]) for row in metadata}) == 8


def test_train_transaction_stays_one_full_192_case_forward(monkeypatch):
    calls = []

    def fake_audit(_model, batch, _cfg, metadata, source):
        calls.append((source, int(batch["clean"].shape[0]), len(metadata)))
        return {"model_forward_calls": 1}

    monkeypatch.setattr(s, "audit_batch", fake_audit)
    batch = {"clean": torch.zeros(192, 1, 151)}
    result = s.audit_train_transaction_0(None, batch, None, [{}] * 192)
    assert result["model_forward_calls"] == 1
    assert calls == [("train", 192, 192)]


def test_final_replays_eight_historical_chunks_with_stable_case_mapping(monkeypatch):
    metadata = []
    seam = torch.zeros(64, 32, 1)
    offset = 0
    for split, role in s.FINAL_BLOCK_ORDER:
        for bank_case_index in range(16):
            width = 10 if bank_case_index % 2 == 0 else 28
            seam[offset + bank_case_index, :width] = 1
            metadata.append({
                "split": split,
                "role": role,
                "width": width,
                "group": f"{role}/{width}",
                "case_index": bank_case_index // 2,
                "bank_case_index": bank_case_index,
            })
        offset += 16
    calls = []

    def fake_audit(_model, batch, _cfg, chunk_metadata, source):
        calls.append((
            source,
            len(chunk_metadata),
            chunk_metadata[0]["split"],
            chunk_metadata[0]["role"],
            chunk_metadata[0]["bank_case_index"],
        ))
        scopes = s._scope_masks(chunk_metadata, "final")
        derivatives = {
            key: {
                objective: {scope: 1.0 for scope in scopes}
                for objective in ("temporal", "endpoint")
            }
            for key in s.ALPHA_KEYS
        }
        return {
            "case_level": [dict(row) for row in chunk_metadata],
            "derivatives": derivatives,
            "alpha_one_parity": {
                "prediction_max_abs_error": 0.0,
                "clean_prediction_max_abs_error": 0.0,
                "rtol": 0.0,
                "atol": s.PARITY_ATOL_CPU,
            },
            "alpha_zero_contract": {
                "contact_channels_unchanged": True,
                "scaled_geometry_matches_alpha": True,
                "geometric_applied_abs_max": 0.0,
            },
            "model_forward_calls": 1,
        }

    monkeypatch.setattr(s, "audit_batch", fake_audit)
    result = s.audit_final_in_historical_chunks(
        None, {"clean": torch.zeros(64, 32, 151), "seam": seam}, None, metadata
    )
    assert len(calls) == result["model_forward_calls"] == 8
    assert all(source == "final" and count == 8 for source, count, *_ in calls)
    assert [(split, role, start) for _, _, split, role, start in calls] == [
        (split, role, start)
        for split, role in s.FINAL_BLOCK_ORDER
        for start in (0, 8)
    ]
    assert len(result["case_level"]) == 64
    assert result["historical_final_replay"] == {
        "chunk_size": 8,
        "chunks": 8,
        "cases": 64,
        "block_order": [f"{split}/{role}" for split, role in s.FINAL_BLOCK_ORDER],
        "one_forward_per_chunk": True,
        "alphas_share_fixed_raw_action_per_chunk": True,
    }


def test_scope_masks_cover_split_role_width_and_group():
    metadata = [{"split": split, "role": role, "width": width}
                for split in ("seen", "new_position")
                for role in ("single_recording", "cross_event") for width in (10, 28)]
    scopes = s._scope_masks(metadata, "final")
    assert int(scopes["overall"].sum()) == 8
    assert int(scopes["split:seen"].sum()) == 4
    assert int(scopes["role:cross_event"].sum()) == 4
    assert int(scopes["width:28"].sum()) == 4
    assert int(scopes["group:new_position/single_recording/28"].sum()) == 1


def test_alignment_lineage_accepts_exact_reviewed_report(tmp_path):
    path = tmp_path / "alignment.json"
    payload = alignment_payload({"report.json": "a"}, "state", {"probe_bank.pt": "p"}, "p")
    path.write_text(json.dumps(payload), encoding="utf-8")
    _, metadata = s.load_alignment_report(path, {"report.json": "a"},
        {"final_state_sha256": "state"}, {"probe_bank.pt": "p"}, "p")
    assert metadata["schema"] == alignment.SCHEMA and metadata["runtime_commit"] == s.REVIEWED_ALIGNMENT_COMMIT


@pytest.mark.parametrize("mutation", ["schema", "runtime", "optimizer", "state", "source",
                                       "trajectory", "probe", "answer", "outside"])
def test_alignment_provenance_or_scientific_premise_mismatch_fails_closed(tmp_path, mutation):
    path = tmp_path / "alignment.json"
    payload = alignment_payload({"report.json": "a"}, "state", {"probe_bank.pt": "p"}, "p")
    if mutation == "schema": payload["schema"] = "wrong"
    elif mutation == "runtime": payload["provenance"]["runtime_commit"] = "wrong"
    elif mutation == "optimizer": payload["optimizer_steps"] = 1
    elif mutation == "state": payload["provenance"]["trajectory_final_state_sha256"] = "wrong"
    elif mutation == "source": payload["provenance"]["source_sha256_including_probe"] = {}
    elif mutation == "trajectory": payload["provenance"]["trajectory_sha256"] = {}
    elif mutation == "probe": payload["provenance"]["probe_sha256"] = "wrong"
    elif mutation == "answer": payload["scientific_answers"]["train_temporal_gradient_present"]["answer"] = False
    else: payload["scientific_answers"]["gradient_outside_decoder_support"]["all_exact_zero"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        s.load_alignment_report(path, {"report.json": "a"}, {"final_state_sha256": "state"},
                                {"probe_bank.pt": "p"}, "p")


def test_alpha_one_final_metric_parity_checks_all_64_cases():
    actual, trajectory = parity_fixture()
    result = s.validate_alpha_one_final_metrics(actual, trajectory)
    assert result["verified"] and result["cases"] == 64 and result["max_absolute_metric_error"] == 0


def test_alpha_one_final_metric_mismatch_fails_closed():
    actual, trajectory = parity_fixture()
    actual[0]["responses"]["1.00"]["authoritative_observable"]["temporal_metric"] = 9
    with pytest.raises(RuntimeError) as error:
        s.validate_alpha_one_final_metrics(actual, trajectory)
    message = str(error.value)
    for field in (
        "split=seen",
        "role=single_recording",
        "width=10",
        "bank_case_index=0",
        "metric=temporal_metric",
        "actual=9",
        "expected=2.0",
        "absolute_error=7.0",
    ):
        assert field in message


def test_actual_cpu_response_uses_one_forward_and_restores_state_mode_grad_hooks():
    cfg, batch = small_batch(cases=1)
    model = m.ProductManifoldTemporalRefiner(hidden=4)
    with torch.no_grad():
        model.out.weight.normal_(0, 1e-4)
    model.train()
    model.in_proj.weight.grad = torch.ones_like(model.in_proj.weight)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    hooks = len(model.out._forward_hooks)
    metadata = [{"split": "train", "role": "single", "width": 10,
                 "group": "single_short", "case_index": 0}]
    with failure.preserve_model_runtime(model), torch.enable_grad():
        result = s.audit_batch(model, batch, cfg, metadata, "train")
    assert result["model_forward_calls"] == 1 and len(result["case_level"][0]["responses"]) == 7
    assert tuple(result["case_level"][0]["responses"]) == s.ALPHA_KEYS
    assert result["alpha_zero_contract"]["geometric_applied_abs_max"] == 0
    assert result["alpha_one_parity"]["prediction_max_abs_error"] == 0
    assert model.training and len(model.out._forward_hooks) == hooks
    assert torch.equal(model.in_proj.weight.grad, torch.ones_like(model.in_proj.weight))
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())


def test_output_json_is_create_only(tmp_path):
    output = tmp_path / "report.json"
    failure._exclusive_json(output, {"schema": s.SCHEMA})
    with pytest.raises(FileExistsError):
        failure._exclusive_json(output, {"schema": s.SCHEMA})


def test_source_has_no_optimizer_parameter_update_or_scale_selection_logic():
    source = inspect.getsource(s)
    assert "torch.optim" not in source and ".backward(" not in source and ".step(" not in source
    assert "scale_selection_performed\": False" in source
    assert ("arg" + "min") not in source.lower()
    assert ("best" + "_alpha") not in source.lower()


def test_report_contract_keeps_pilot_publication_and_acceptance_false():
    source = inspect.getsource(s)
    assert "scientific_acceptance\": False" in source
    assert "publish_allowed\": False" in source
    assert "pilot_allowed\": False" in source
    assert "optimizer_steps\": 0" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_scale_and_decoder_smoke():
    cfg, batch = small_batch(cases=1)
    cfg.device = "cuda"
    batch = {key: value.cuda() for key, value in batch.items()}
    model = m.ProductManifoldTemporalRefiner(hidden=4).cuda()
    with torch.no_grad():
        model.out.weight.normal_(0, 1e-4)
    raw = s.capture_fixed_action(model, batch, cfg)
    output = s.scale_raw_output(raw["repair"], 1.25)
    trace = {}
    masks = m._refiner_decode_masks(batch["joint"], batch["root"], batch["contact"], batch["seam"], cfg)
    prediction = m._decode_product_refiner_output(batch["bad"], output, *masks, cfg, trace=trace)
    assert prediction.is_cuda and set(s.TRACE_STAGES) <= set(trace)
