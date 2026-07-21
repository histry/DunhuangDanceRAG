"""Canonical SMPL24/EDGE151 skeleton and channel contract.

This module is the single source of truth for every FK, retargeting, training,
audit, rendering and scheduling path.  It is intentionally NumPy-only so that
loading the contract never depends on a training runtime.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

SMPL24_SKELETON_SCHEMA = "dunhuang_smpl24_canonical_v1"
# Serialized identifier shared with motion_geometry.rotations.
ROT6D_LAYOUT = "column"

CONTACT = slice(0, 4)
ROOT = slice(4, 7)
ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX = 4, 5, 6
ROT6D_START, ROT6D_END = 7, 151
MOTION_DIM = 151
NUM_JOINTS = 24
FOOT_JOINTS = (7, 8, 10, 11)

JOINT_NAMES = (
    "root", "lhip", "rhip", "belly", "lknee", "rknee", "spine",
    "lankle", "rankle", "chest", "ltoes", "rtoes", "neck",
    "linshoulder", "rinshoulder", "head", "lshoulder", "rshoulder",
    "lelbow", "relbow", "lwrist", "rwrist", "lhand", "rhand",
)

PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
     9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)

OFFSETS = np.asarray(
    [
        [0.00000000, 0.00000000, 0.00000000],
        [0.05858135, -0.08228004, -0.01766408],
        [-0.06030973, -0.09051332, -0.01354254],
        [0.00443945, 0.12440352, -0.03838522],
        [0.04345142, -0.38646945, 0.00803700],
        [-0.04325663, -0.38368791, -0.00484304],
        [0.00448844, 0.13795640, 0.02682033],
        [-0.01479032, -0.42687458, -0.03742800],
        [0.01905555, -0.42004550, -0.03456167],
        [-0.00226458, 0.05603239, 0.00285505],
        [0.04105436, -0.06028581, 0.12204243],
        [-0.03483987, -0.06210566, 0.13032329],
        [-0.01339020, 0.21163553, -0.03346758],
        [0.07170245, 0.11399969, -0.01889817],
        [-0.08295366, 0.11247234, -0.02370739],
        [0.01011321, 0.08893734, 0.05040987],
        [0.12292141, 0.04520509, -0.01904600],
        [-0.11322832, 0.04685326, -0.00847207],
        [0.25533190, -0.01564902, -0.02294649],
        [-0.26012748, -0.01436928, -0.03126873],
        [0.26570925, 0.01269811, -0.00737473],
        [-0.26910836, 0.00679372, -0.00602676],
        [0.08669055, -0.01063603, -0.01559429],
        [-0.08875370, -0.00865157, -0.01010708],
    ],
    dtype=np.float32,
)

# Compatibility names.  They alias the same immutable contract arrays.
SMPL_JOINT_NAMES = JOINT_NAMES
SMPL_PARENTS = PARENTS
SMPL_OFFSETS = OFFSETS


def skeleton_fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(SMPL24_SKELETON_SCHEMA.encode("ascii"))
    digest.update(ROT6D_LAYOUT.encode("ascii"))
    digest.update("\n".join(JOINT_NAMES).encode("utf-8"))
    digest.update(PARENTS.astype("<i8", copy=False).tobytes())
    digest.update(OFFSETS.astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def skeleton_contract() -> dict[str, Any]:
    return {
        "schema": SMPL24_SKELETON_SCHEMA,
        "num_joints": NUM_JOINTS,
        "motion_dim": MOTION_DIM,
        "rot6d_layout": ROT6D_LAYOUT,
        "joint_names": list(JOINT_NAMES),
        "parents": PARENTS.tolist(),
        "offsets_m": OFFSETS.tolist(),
        "sha256": skeleton_fingerprint(),
    }


def skeleton_contract_json() -> str:
    return json.dumps(skeleton_contract(), ensure_ascii=True, sort_keys=True)


def validate_skeleton_contract() -> None:
    if len(JOINT_NAMES) != NUM_JOINTS:
        raise RuntimeError("SMPL24 joint-name count mismatch")
    if PARENTS.shape != (NUM_JOINTS,) or OFFSETS.shape != (NUM_JOINTS, 3):
        raise RuntimeError("SMPL24 parent/offset shape mismatch")
    if int(PARENTS[0]) != -1 or np.any(PARENTS[1:] >= np.arange(1, NUM_JOINTS)):
        raise RuntimeError("SMPL24 parent order is not topological")
    if not np.isfinite(OFFSETS).all():
        raise RuntimeError("SMPL24 offsets contain NaN/Inf")


validate_skeleton_contract()
