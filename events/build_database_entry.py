#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the formal SMPL Event-DB with physical endpoint enrichment."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from events import build_database as base  # noqa: E402
from events.filter_anatomy import filter_database  # noqa: E402
from events.intrinsic_geometry import augment_database  # noqa: E402


def _arg_value(args: Sequence[str], flag: str) -> Optional[str]:
    try:
        index = list(args).index(flag)
        return str(args[index + 1])
    except (ValueError, IndexError):
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the only formal Event-DB construction chain.

    Local posture states and intrinsic endpoint geometry are project-owned
    physical features. They are required before AESD and Scheduler indexing and
    do not reintroduce the archived BVH, external Grounder, or old retrieval
    business paths.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    result = int(base.main(args) or 0)
    if result != 0:
        return result

    out_db = _arg_value(args, "--out_db")
    if not out_db:
        raise RuntimeError("Formal Event-DB builder requires --out_db")
    out_dir = Path(out_db)
    db_path = out_dir / "events.npz"
    meta_path = out_dir / "events_meta.json"
    if not db_path.is_file():
        raise FileNotFoundError(str(db_path))

    anatomy_audit_path = out_dir / "events.anatomy.audit.json"
    geometry_audit_path = out_dir / "events.intrinsic_geometry.audit.json"
    anatomy_report = filter_database(
        db_path,
        meta_path,
        anatomy_audit_path,
    )
    geometry_report = augment_database(
        db_path,
        geometry_audit_path,
        fps=None,
    )
    report = {
        "schema": "formal_smpl_event_db_physical_enrichment_v1",
        "db": str(db_path),
        "anatomy": {
            "schema": anatomy_report.get("schema"),
            "events_before": anatomy_report.get("events_before"),
            "events_after": anatomy_report.get("events_after"),
            "audit": str(anatomy_audit_path),
            "ok": bool(anatomy_report.get("ok", False)),
        },
        "intrinsic_geometry": {
            "schema": geometry_report.get("schema"),
            "num_events": geometry_report.get("num_events"),
            "fps": geometry_report.get("fps"),
            "quality": geometry_report.get("quality"),
            "audit": str(geometry_audit_path),
            "ok": bool(geometry_report.get("ok", False)),
        },
        "archived_business_paths_restored": False,
        "ok": bool(
            anatomy_report.get("ok", False) and geometry_report.get("ok", False)
        ),
    }
    report_path = out_dir / "events.formal_build.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not report["ok"]:
        raise RuntimeError(f"Formal Event-DB physical enrichment failed: {report_path}")
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "events_after_anatomy": anatomy_report.get("events_after"),
                "geometry_events": geometry_report.get("num_events"),
                "report": str(report_path),
                "ok": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
