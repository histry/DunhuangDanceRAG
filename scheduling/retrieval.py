"""Candidate retrieval primitives used by whole-song schedulers."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def precompute_music_similarity(
    router,
    queries: Sequence[np.ndarray],
    motion_desc: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Return the slot-by-event similarity matrix for learned or rule routing."""
    query_matrix = np.stack(queries).astype(np.float32)
    descriptors = np.asarray(motion_desc, dtype=np.float32)
    if router is None:
        distance = np.linalg.norm(
            query_matrix[:, None, :] - descriptors[None, :, :],
            axis=-1,
        )
        return (1.0 - distance / np.sqrt(descriptors.shape[1])).astype(np.float32)

    with torch.no_grad():
        query_tensor = torch.from_numpy(query_matrix).to(device)
        descriptor_tensor = torch.from_numpy(descriptors).to(device)
        query_embedding = router.encode_music(query_tensor)
        motion_embedding = router.encode_motion(descriptor_tensor)
        similarity = query_embedding @ motion_embedding.t()
    return similarity.detach().cpu().numpy().astype(np.float32)
