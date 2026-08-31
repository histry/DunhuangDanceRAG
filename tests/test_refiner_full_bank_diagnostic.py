import inspect

import pytest
import torch

from training import refiner_bridge_diagnostics as d


@pytest.mark.parametrize('width,recipe_id', [(10,0),(28,1)])
def test_fit_context_starts_are_deterministic_and_exclude_probe_guard(width,recipe_id):
    seen,probe=d._seen_and_probe_starts(120,width,recipe_id)
    starts=d._context_fit_starts(120,width,recipe_id)
    assert len(starts)==d.FIT_CONTEXT_COUNT
    assert len(set(starts))==len(starts)
    assert seen not in starts and probe not in starts
    assert all(abs(start-probe)>d.PROBE_START_GUARD_FRAMES for start in starts)
    assert starts==d._context_fit_starts(120,width,recipe_id)


def test_anchored_context_replay_uses_all_seen_cases_and_never_reads_probe():
    class SeenOnly(dict):
        def __getitem__(self, key):
            assert key[0] != 'new_position', 'held-out position leaked into fitting'
            return super().__getitem__(key)
    def role(offset):
        x=torch.arange(offset,offset+16)[:,None,None].float()
        return {'clean':x,'bad':x+100,'cond':x+200,'clean_cond':x+300}
    banks=SeenOnly({('seen','single_recording'):role(0),
                    ('seen','cross_event'):role(16)})
    for context in range(d.FIT_CONTEXT_COUNT):
        banks[(f'fit_context_{context}','single_recording')]=role(32+32*context)
        banks[(f'fit_context_{context}','cross_event')]=role(48+32*context)
    batches=d.anchored_context_replay_banks(banks)
    assert len(batches)==1
    batch=batches[0]
    expected_cases = 4 * 8 * (1 + d.FIT_CONTEXT_COUNT)
    torch.testing.assert_close(
        batch['clean'].flatten(),
        torch.arange(expected_cases).float(),
    )
    expected_per_group = 8 * (1 + d.FIT_CONTEXT_COUNT)
    assert torch.bincount(batch['group']).tolist() == [expected_per_group] * 4
    assert len(batch['clean_cond']) == expected_cases

def test_fixed_bank_rejects_missing_or_unpaired_role_cases():
    a={'clean':torch.zeros(16,1,1)}
    for count in (15,14):
        with pytest.raises(ValueError,match='paired'):
            d.fixed_fit_bank({('seen','single_recording'):a,
                              ('seen','cross_event'):{'clean':torch.zeros(count,1,1)}})


def test_fit_contract_counts_examples_not_just_iterations():
    import inspect

    contract = d.fit_bank_contract(8)

    # Transaction size is frozen:
    # 32 seen + 5*32 context = 192.
    assert contract["cases_per_update"] == 192
    assert contract["seen_anchor_cases_per_update"] == 32
    assert contract["context_cases_per_update"] == 160
    assert contract["cases_per_role_width"] == 48
    assert contract["cases_per_role_width_per_bank"] == 8

    # C5 remains PER UPDATE; the reservoir itself is larger.
    assert contract["context_banks_per_update"] == d.FIT_CONTEXT_COUNT
    assert contract["context_reservoir_cycle_length"] > d.FIT_CONTEXT_COUNT

    assert (
        contract["context_reservoir_protocol"]
        == d.CONTEXT_RESERVOIR_PROTOCOL
    )

    assert (
        contract["gradient_scope"]
        == "complete_seen_plus_rotating_c5_safe_context_reservoir"
    )

    assert (
        contract["line_search_scope"]
        == "same_complete_seen_plus_rotating_c5_transaction"
    )

    assert (
        contract["transaction_batch_fixed_within_step"]
        is True
    )

    assert (
        contract["reservoir_every_legal_start_seen_per_cycle"]
        is True
    )

    assert contract["probe_start_guard_frames"] == 6
    assert contract["probe_used_for_updates"] is False

    source = inspect.getsource(d.run)
    compact = "".join(source.split())

    assert "balanced_indices(" not in source

    # V15.4.1 must NOT materialize the complete transaction cycle.
    assert (
        "anchored_context_replay_banks(banks)"
        not in source
    )

    assert "train_cycle[" not in source

    # One C5 transaction is selected and materialized for this step only.
    assert (
        "selected_context_indices="
        in compact
    )

    assert (
        "batch=_reservoir_transaction_batch("
        in compact
    )

    # Armijo closure still uses the SAME local batch object.
    assert (
        "model,batch,cfg,require_all_groups=True"
        in compact
    )

    assert (
        "include_fit_contexts=not"
        'getattr(args,"baseline_only",False)'
        in compact
    )

    assert (
        "fit_bank_contract(args.windows,cfg)"
        in compact
    )

    assert (
        d.PROBE_SCOPE
        == "unfitted_local_motion_context_within_train_windows"
    )

    assert (
        '"probe_scope":PROBE_SCOPE'
        in compact
    )

def test_fixed_bank_stall_is_not_counted_as_400_steps_or_pilot_acceptance():
    for reason in ('bounded_search_no_descent','zero_gradient'):
        assert d.fixed_bank_stalled({'optimizer_update_accepted':False,'reason':reason})
    assert not d.fixed_bank_stalled({'optimizer_update_accepted':True,
                                     'reason':'same_batch_loss_decreased'})
    from training import motion_models as m
    assert 'fixed_bank_stalled' not in inspect.getsource(m.train_refiner)


def test_stalled_report_cannot_authorize_pilot(tmp_path,monkeypatch):
    import json
    from argparse import Namespace
    report=tmp_path/'stalled.json'
    report.write_text(json.dumps({'schema':d.SCHEMA,'fingerprint':{},'completed':True,
        'published':False,'stopped_early':True,'completed_steps':150,'target_steps':400}))
    monkeypatch.setattr(d,'fingerprint',lambda *args:{})
    with pytest.raises(RuntimeError,match='optimization stalled'):
        d.run(Namespace(config='configs/motion_model.json',check_report=str(report)))


def test_complete_step_count_without_complete_fit_bank_cannot_authorize_pilot(tmp_path,monkeypatch):
    import json
    from argparse import Namespace
    from training.refiner_optimizer import REFINER_UPDATE_PROTOCOL
    report=tmp_path/'subset.json'
    report.write_text(json.dumps({'schema':d.SCHEMA,'fingerprint':{},'completed':True,
        'published':False,'target_steps':400,'completed_steps':400,'windows':[{}]*8,
        'fit_bank':{**d.fit_bank_contract(8),'cases_per_update':8},
        'optimizer_updates':{'protocol':REFINER_UPDATE_PROTOCOL,'attempted_steps':400,
            'accepted_steps':400,'retained_steps':0,'trial_evaluations':400,
            'accepted_non_descent_steps':0}}))
    monkeypatch.setattr(d,'fingerprint',lambda *args:{})
    with pytest.raises(RuntimeError,match='complete predefined TRAIN context cycle'):
        d.run(Namespace(config='configs/motion_model.json',check_report=str(report),windows=8))


def test_portable_bank_and_optimizer_state_are_diagnostic_only(tmp_path):
    from training import motion_models as m

    model = torch.nn.Linear(
        1,
        1,
    )

    optimizer = torch.optim.AdamW(
        model.parameters()
    )

    model(
        torch.ones(1, 1)
    ).sum().backward()

    optimizer.step()

    cfg = m.MotionGenerationConfig()

    contract = d.fit_bank_contract(
        8,
        cfg,
    )

    reservoir_count = contract[
        "context_reservoir_cycle_length"
    ]

    def role(value):
        x = torch.full(
            (16, 1, 1),
            float(value),
        )

        return {
            "clean": x,
        }

    banks = {
        ("seen", "single_recording"):
            role(-2),

        ("seen", "cross_event"):
            role(-1),
    }

    for index in range(
        reservoir_count
    ):
        banks[
            (
                f"fit_context_{index}",
                "single_recording",
            )
        ] = role(
            2 * index
        )

        banks[
            (
                f"fit_context_{index}",
                "cross_event",
            )
        ] = role(
            2 * index + 1
        )

    schedule = (
        d._reservoir_transaction_schedule(
            banks
        )
    )

    assert len(schedule) == reservoir_count

    report = {
        "fingerprint": {
            "test": "exact",
        },

        "windows": [],

        "fit_bank":
            contract,
    }

    # V15.4.1 stores unique reservoir banks + deterministic schedule.
    # It deliberately does NOT serialize eagerly concatenated transactions.
    report["fit_bank_artifact"] = (
        d.save_fit_bank(
            tmp_path,
            report,
            cfg,
            banks=banks,
            schedule=schedule,
        )
    )

    probe_batch = {
        "clean":
            torch.zeros(
                16,
                1,
                1,
            ),

        "bad":
            torch.ones(
                16,
                1,
                1,
            ),
    }

    probe_banks = {
        (
            "new_position",
            role_name,
        ): probe_batch
        for role_name in (
            "single_recording",
            "cross_event",
        )
    }

    report["probe_bank_artifact"] = (
        d.save_probe_bank(
            tmp_path,
            probe_banks,
            report,
            cfg,
        )
    )

    d.save_diagnostic_state(
        tmp_path,
        model,
        optimizer,
        report,
        19,
    )

    bank = m._trusted_torch_load(
        tmp_path / "fit_bank.pt",
        map_location="cpu",
    )

    probe = m._trusted_torch_load(
        tmp_path / "probe_bank.pt",
        map_location="cpu",
    )

    state = m._trusted_torch_load(
        tmp_path / "diagnostic_state.pt",
        map_location="cpu",
    )

    assert bank["train_only"]
    assert not bank["formal_checkpoint"]
    assert not bank["publish_allowed"]

    assert (
        bank["schema"]
        == "refiner_train_safe_start_context_reservoir_v4"
    )

    assert set(bank) >= {
        "anchor",
        "context_reservoir",
        "transaction_schedule",
    }

    # The exact reservoir is stored once.
    assert (
        len(bank["context_reservoir"])
        == reservoir_count
    )

    # The exact deterministic transaction sequence is replayable.
    assert (
        len(bank["transaction_schedule"])
        == reservoir_count
    )

    assert (
        tuple(
            tuple(int(i) for i in row)
            for row in bank[
                "transaction_schedule"
            ]
        )
        == tuple(schedule)
    )

    # No old eager transaction list is serialized.
    assert "batches" not in bank

    assert (
        bank["anchor"]["clean"].device.type
        == "cpu"
    )

    assert all(
        row["clean"].device.type == "cpu"
        for row in bank[
            "context_reservoir"
        ].values()
    )

    assert probe["probe_only"]
    assert probe["updates_forbidden"]
    assert not probe["formal_checkpoint"]
    assert not probe["publish_allowed"]

    assert set(
        probe["banks"]
    ) == {
        "single_recording",
        "cross_event",
    }

    assert (
        state["completed_steps"]
        == 19
    )

    assert (
        report[
            "fit_bank_artifact"
        ]["reservoir_banks"]
        == reservoir_count
    )

    assert (
        report[
            "fit_bank_artifact"
        ]["transactions_per_cycle"]
        == reservoir_count
    )

    assert (
        report[
            "fit_bank_artifact"
        ]["sha256"]
        == d.common.file_sha256(
            tmp_path / "fit_bank.pt"
        )
    )

    assert (
        report[
            "probe_bank_artifact"
        ]["sha256"]
        == d.common.file_sha256(
            tmp_path / "probe_bank.pt"
        )
    )

    assert (
        state["probe_bank_artifact"]
        == report[
            "probe_bank_artifact"
        ]
    )

    assert state[
        "optimizer_state_dict"
    ]["state"]

    assert "training_resume" not in state
    assert "version" not in state


def test_unlogged_stall_records_gradients_exact_state_and_return_code(
    tmp_path,
    monkeypatch,
):
    import json
    import numpy as np

    from argparse import Namespace

    from training import motion_models as m
    from training import bridge_feasibility as f
    from training.refiner_optimizer import REFINER_UPDATE_PROTOCOL

    # The old 8-frame synthetic window predates the +/-6 probe guard and
    # cannot physically contain a legal C5 reservoir. Use the real diagnostic
    # window length while keeping all tensors synthetic and CPU-only.
    cfg = m.MotionGenerationConfig(
        device="cpu",
        window_len=120,
    )

    reservoir_count = (
        d._context_reservoir_cycle_length(
            cfg.window_len
        )
    )

    assert reservoir_count > d.FIT_CONTEXT_COUNT

    monkeypatch.setattr(
        m.MotionGenerationConfig,
        "from_json",
        lambda path: cfg,
    )

    monkeypatch.setattr(
        m.MotionGenerationConfig,
        "apply_env",
        lambda self: self,
    )

    db = {
        "paths":
            np.array(
                ["ignored"] * 8
            ),

        "source_uids":
            np.array(
                [str(i) for i in range(8)]
            ),

        "source_formats":
            np.array(
                ["chang_e_official_smpl"] * 8
            ),
    }

    monkeypatch.setattr(
        m,
        "load_db",
        lambda path: db,
    )

    monkeypatch.setattr(
        m,
        "_training_db_contract",
        lambda *a: {},
    )

    monkeypatch.setattr(
        m,
        "_validate_source_disjoint",
        lambda *a: {
            "verified": True,
        },
    )

    monkeypatch.setattr(
        m,
        "load_motion_window",
        lambda *a, **k:
            np.zeros(
                (
                    cfg.window_len,
                    151,
                )
            ),
    )

    monkeypatch.setattr(
        m,
        "_descriptor_values_in_training_coordinates",
        lambda *a:
            np.zeros(
                (
                    8,
                    32,
                )
            ),
    )

    splits = (
        "seen",
        "new_position",
        *(
            f"fit_context_{i}"
            for i in range(
                reservoir_count
            )
        ),
    )

    banks = {
        (
            split,
            role,
        ): {
            "clean":
                torch.zeros(
                    16,
                    1,
                    1,
                )
        }
        for split in splits
        for role in (
            "single_recording",
            "cross_event",
        )
    }

    monkeypatch.setattr(
        d,
        "build_banks",
        lambda *a, **k:
            (
                banks,
                {},
            ),
    )

    monkeypatch.setattr(
        d,
        "fingerprint",
        lambda *a: {},
    )

    monkeypatch.setattr(
        d.common,
        "file_sha256",
        lambda path: "test-digest",
    )

    monkeypatch.setattr(
        f,
        "check_foundation_report",
        lambda *a: None,
    )

    monkeypatch.setattr(
        f,
        "group_decisions",
        lambda *a: {
            "group": {
                "passed": False,
            }
        },
    )

    monkeypatch.setattr(
        m,
        "ProductManifoldTemporalRefiner",
        lambda **kwargs:
            torch.nn.Linear(
                1,
                1,
            ),
    )

    def objective(
        model,
        batch,
        cfg,
    ):
        r = sum(
            p.square().sum()
            for p in model.parameters()
        )

        terms = {
            "endpoint": r,
        }

        for label in m.REFINER_GROUP_LABELS:
            terms[
                f"group_{label}_repair_total"
            ] = r

            terms[
                f"group_{label}_endpoint_continuity"
            ] = r

            terms[
                f"group_{label}_temporal_supervision_raw"
            ] = r

            terms[
                f"group_{label}_joint_scientific_deficit"
            ] = r

        return (
            r,
            r * 0,
            terms,
            {},
        )

    monkeypatch.setattr(
        m,
        "_refiner_batch_objectives",
        objective,
    )

    monkeypatch.setattr(
        m,
        "_refiner_gradient_diagnostics",
        lambda *a: {
            "recorded": True,
        },
    )

    monkeypatch.setattr(
        m,
        "_refiner_component_gradients",
        lambda *a: {
            "recorded": True,
        },
    )

    calls = []

    def step(*a, **k):
        calls.append(1)

        accepted = len(calls) == 1

        return {
            "protocol":
                REFINER_UPDATE_PROTOCOL,

            "optimizer_update_accepted":
                accepted,

            "reason":
                (
                    "same_batch_loss_decreased"
                    if accepted
                    else "bounded_search_no_descent"
                ),

            "used_gradient_rescue":
                not accepted,

            "trial_evaluations":
                1,

            "nonfinite_trials":
                0,

            "insufficient_decrease_trials":
                0,

            "group_guard_rejected_trials":
                0,

            "accepted_non_descent_steps":
                0,

            "loss_before":
                1.0,

            "loss_after":
                (
                    0.9
                    if accepted
                    else 1.0
                ),
        }

    monkeypatch.setattr(
        m,
        "checked_refiner_step",
        step,
    )

    monkeypatch.setattr(
        d,
        "evaluate",
        lambda *a: {},
    )

    monkeypatch.setattr(
        d,
        "failure_breakdown",
        lambda *a: {},
    )

    monkeypatch.setattr(
        m,
        "_checkpoint_validation_decision",
        lambda *a, **k: {
            "scientific_acceptance":
                False,

            "reasons":
                ["not_ready"],

            "observed":
                {},
        },
    )

    out = tmp_path / "diagnostic"

    result = d.run(
        Namespace(
            config="unused",
            check_report=None,
            out_dir=str(out),
            steps=400,
            eval_every=200,
            windows=8,
            foundation_report=str(
                tmp_path / "foundation.json"
            ),
            db="train",
            val_db="val",
        )
    )

    report = json.loads(
        (
            out
            / "diagnostic_report.json"
        ).read_text()
    )

    logs = [
        json.loads(row)
        for row in (
            out
            / "gradients.jsonl"
        ).read_text().splitlines()
    ]

    state = m._trusted_torch_load(
        out / "diagnostic_state.pt",
        map_location="cpu",
    )

    # First transaction is accepted. Then a FULL reservoir cycle must stall
    # before early termination is allowed.
    expected_steps = (
        1
        + reservoir_count
    )

    assert result == 2
    assert len(calls) == expected_steps

    assert report["stopped_early"]
    assert report["completed_steps"] == expected_steps
    assert not report["diagnostic_ready"]

    assert logs[-1]["step"] == expected_steps
    assert logs[-1]["gradient"]["recorded"]
    assert logs[-1]["component_gradients"]["recorded"]

    assert (
        logs[-1]["fit_reservoir_cycle_length"]
        == reservoir_count
    )

    assert state[
        "completed_steps"
    ] == expected_steps

    assert not state[
        "formal_checkpoint"
    ]

    assert len(
        (
            out
            / "optimizer_updates.jsonl"
        ).read_text().splitlines()
    ) == expected_steps
