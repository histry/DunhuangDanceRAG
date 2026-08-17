from pathlib import Path

from data_pipeline.chang_e_manifest import load_manifest, validate_source
from data_pipeline.split_sources import recording_group_records
from training.motion_models import parse_change_bvh_semantics


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "assets" / "motion" / "bvh"


def test_manifest_overrides_incorrect_bvh_header_timebase():
    report = validate_source(SOURCE_DIR / "female_36pose_1.bvh")

    assert round(report["declared_fps"], 3) == 24.0
    assert report["effective_fps"] == 60.0
    assert 327.0 < report["duration_seconds"] < 330.0


def test_synchronized_dancers_share_recording_but_not_source_identity():
    first = parse_change_bvh_semantics(SOURCE_DIR / "female_36pose_1.bvh")
    second = parse_change_bvh_semantics(SOURCE_DIR / "female_36pose_2.bvh")

    assert first["source_uid"] != second["source_uid"]
    assert first["recording_uid"] == second["recording_uid"]
    assert first["take_id"] == -1
    assert second["take_id"] == -1


def test_local_inventory_collapses_to_nine_recording_split_units():
    manifest = load_manifest()
    rows = [
        {
            "source_uid": row["source_id"],
            "recording_uid": row["recording_uid"],
            "performer_group": row["performer_group"],
            "dance_key": row["dance_category"],
        }
        for row in manifest["sources"]
    ]

    units = recording_group_records(rows)

    assert len(rows) == 12
    assert len(units) == 9
    paired = {
        unit["recording_uid"]: unit["num_performer_tracks"] for unit in units
    }
    assert paired["female_36pose_sequence"] == 2
    assert paired["male_36pose_sequence"] == 2
    assert paired["male_drum_sequence"] == 2
