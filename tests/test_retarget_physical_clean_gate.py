from retargeting.build_cache import _report_valid


def _report(physical_clean_ok: bool):
    return {
        "version": "official_smpl_event_geometry_1",
        "ok": True,
        "source_gate_ok": True,
        "gravity_ok": True,
        "fit_ok": True,
        "physical_clean_ok": physical_clean_ok,
    }


def test_retarget_cache_requires_pretraining_physical_clean_gate():
    valid, reasons = _report_valid(_report(True))
    assert valid is True
    assert reasons == []

    valid, reasons = _report_valid(_report(False))
    assert valid is False
    assert reasons == ["physical_clean_not_ok"]
