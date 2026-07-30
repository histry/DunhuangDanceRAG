import io

from routing.anatomy_feature_cache import AnatomyProgressMonitor


def test_progress_monitor_emits_auditable_rows(monkeypatch):
    monkeypatch.setenv("V46_52_ANATOMY_PROGRESS_ENABLE", "1")
    stream = io.StringIO()
    monitor = AnatomyProgressMonitor(stream=stream)
    monitor.start(3, {"beam": 4, "topk": 8})
    monitor.slot_start(0, 90, 1, 8)
    token = monitor.candidate_start(
        slot=0,
        state_index=0,
        event_id=12,
        candidate_rank=2,
        target_frames=90,
    )
    monitor.candidate_finish(token, safe=True)
    monitor.slot_finish(0, 4, 2)
    monitor.finish()
    text = stream.getvalue()
    assert "[ANATOMY-PROGRESS]" in text
    assert '"event_id": 12' in text
    assert '"event": "slot_complete"' in text
