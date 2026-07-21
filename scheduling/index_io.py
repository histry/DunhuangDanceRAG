"""Validated I/O for the shared scheduler event index."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from motion_geometry.rotations import (
    CANONICAL_ROT6D_LAYOUT,
    normalize_rot6d_layout,
)
from support.event_identity import (
    event_uid_from_item,
    make_event_db_contract,
    normalize_event_db_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_EVENT_ARRAYS = (
    "motion_desc",
    "mmr_embed",
    "entry_pose",
    "exit_pose",
    "entry_vel",
    "exit_vel",
    "length",
)


def event_motion_reference(item: Dict[str, Any]) -> str:
    value = item.get("pkl", item.get("path", ""))
    if value is None or not str(value).strip():
        raise ValueError("Event item has neither a non-empty 'pkl' nor 'path' field")
    return str(value)


def resolve_event_motion_path(
    item_or_reference: Dict[str, Any] | str | Path,
    index_path: str | Path,
    *,
    metadata: Dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve an event motion independently of the process working directory.

    Project-local paths such as ``assets/events/...`` are rooted at the project
    package.  Optional index metadata roots are also supported for portable
    external asset stores.  The current directory is retained only as the last
    backwards-compatible fallback.
    """

    raw_value = (
        event_motion_reference(item_or_reference)
        if isinstance(item_or_reference, dict)
        else str(item_or_reference)
    )
    raw = Path(raw_value).expanduser()
    index = Path(index_path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve() if project_root else PROJECT_ROOT

    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(root / raw)
        candidates.append(index.parent / raw)

        info = metadata or {}
        for key in ("asset_root", "event_root", "motion_root"):
            value = info.get(key)
            if not value:
                continue
            declared_root = Path(str(value)).expanduser()
            if not declared_root.is_absolute():
                declared_root = index.parent / declared_root
            candidates.append(declared_root / raw)

        # Compatibility only; correctness does not depend on this candidate.
        candidates.append(Path.cwd() / raw)

    checked: List[str] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        checked.append(key)
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(
        f"Cannot resolve event motion {raw_value!r} from index {index}; "
        f"checked={checked}"
    )


def load_shared_index(
    json_path: str | Path,
    npz_path: str | Path,
) -> Tuple[Dict[str, Any], Any, List[Dict[str, Any]]]:
    """Load metadata and aligned arrays without changing event order."""
    metadata_path = Path(json_path)
    arrays_path = Path(npz_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("items"), list):
        raise ValueError(f"Event index must contain an items list: {metadata_path}")
    declared_layout = metadata.get("rot6d_layout")
    if declared_layout is None:
        raise RuntimeError(
            f"Event index has no rot6d_layout contract: {metadata_path}. "
            "Migrate the asset explicitly instead of guessing its geometry."
        )
    normalized_layout = normalize_rot6d_layout(str(declared_layout))
    if normalized_layout != CANONICAL_ROT6D_LAYOUT:
        raise RuntimeError(
            f"Scheduler requires canonical rot6d_layout={CANONICAL_ROT6D_LAYOUT!r}, "
            f"but index declares {normalized_layout!r}: {metadata_path}"
        )
    metadata["rot6d_layout"] = normalized_layout

    arrays = np.load(arrays_path, allow_pickle=True)
    items = metadata["items"]
    count = len(items)
    missing = [name for name in REQUIRED_EVENT_ARRAYS if name not in arrays.files]
    if missing:
        raise RuntimeError(f"Scheduler index is missing arrays {missing}: {arrays_path}")
    for name in REQUIRED_EVENT_ARRAYS:
        if len(arrays[name]) != count:
            raise RuntimeError(
                f"Index mismatch: {name} has {len(arrays[name])}, metadata has {count}"
            )
    event_uids = [event_uid_from_item(item, position=i) for i, item in enumerate(items)]
    if len(set(event_uids)) != count:
        raise RuntimeError(f"Scheduler index contains duplicate event_uid values: {metadata_path}")
    for item, event_uid in zip(items, event_uids):
        item["event_uid"] = event_uid
        item.setdefault("event_id", event_uid)
    computed_contract = make_event_db_contract(event_uids)
    declared_contract = normalize_event_db_contract(metadata.get("event_db_contract"))
    if declared_contract is not None and declared_contract != computed_contract:
        raise RuntimeError(
            f"Scheduler index event DB fingerprint is stale: declared={declared_contract}, "
            f"computed={computed_contract}"
        )
    metadata["event_db_contract"] = computed_contract
    if "event_uids" in arrays.files:
        stored = [str(value) for value in arrays["event_uids"]]
        if stored != event_uids:
            raise RuntimeError("Scheduler JSON and NPZ event_uid order do not match")
    return metadata, arrays, items
