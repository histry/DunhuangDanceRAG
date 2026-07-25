#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diversity-constrained candidate selection for AESD retrieval.

The selector is a post-ranking operator.  It preserves score order while
limiting repeated ``source_uid``, event family and AESD class.  Class capacities
are derived from the music semantic probability vector rather than from a fixed
uniform quota, so a genuine climax slot can still allocate more turning events.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from events.semantic_descriptor import MUSIC_SEMANTIC_LABELS, normalize_vector


def _class_capacities(
    music_probabilities: Sequence[float], top_k: int, minimum: int = 2
) -> Dict[str, int]:
    probabilities = normalize_vector(np.asarray(music_probabilities, dtype=np.float32))
    raw = np.maximum(
        int(minimum), np.rint(probabilities * int(top_k)).astype(np.int64)
    )
    # Reduce excess capacity from the least likely classes while retaining a
    # small minority-class path.  Capacities may sum above top_k; that is valid
    # because they are upper bounds rather than a required allocation.
    return {
        label: int(raw[index])
        for index, label in enumerate(MUSIC_SEMANTIC_LABELS)
    }


def select_diverse_candidates(
    scores: Sequence[float],
    source_uids: Sequence[Any],
    event_families: Sequence[Any],
    semantic_labels: Sequence[Any],
    music_probabilities: Sequence[float],
    *,
    top_k: int = 20,
    source_cap: int = 3,
    family_cap: int = 6,
    minimum_class_cap: int = 2,
    eligible_mask: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    sources = np.asarray(source_uids, dtype=object).reshape(-1)
    families = np.asarray(event_families, dtype=object).reshape(-1)
    semantics = np.asarray(semantic_labels, dtype=object).reshape(-1)
    if not (len(values) == len(sources) == len(families) == len(semantics)):
        raise ValueError("Candidate metadata is not row-aligned")
    if eligible_mask is None:
        eligible = np.ones(len(values), dtype=bool)
    else:
        eligible = np.asarray(eligible_mask, dtype=bool).reshape(-1)
        if len(eligible) != len(values):
            raise ValueError("eligible_mask has the wrong length")
    capacities = _class_capacities(
        music_probabilities, int(top_k), int(minimum_class_cap)
    )
    source_count: Counter[str] = Counter()
    family_count: Counter[str] = Counter()
    semantic_count: Counter[str] = Counter()
    selected = []
    order = np.argsort(values, kind="stable")[::-1]
    for index in order:
        index = int(index)
        if not eligible[index]:
            continue
        source = str(sources[index])
        family = str(families[index])
        semantic = str(semantics[index])
        if source_count[source] >= int(source_cap):
            continue
        if family_count[family] >= int(family_cap):
            continue
        if semantic_count[semantic] >= capacities.get(
            semantic, int(minimum_class_cap)
        ):
            continue
        selected.append(index)
        source_count[source] += 1
        family_count[family] += 1
        semantic_count[semantic] += 1
        if len(selected) >= int(top_k):
            break
    # Controlled relaxation prevents an empty/undersized shortlist.  The first
    # pass is the scientific diversity contract; the second pass is explicitly
    # auditable and only fills unused positions.
    if len(selected) < int(top_k):
        selected_set = set(selected)
        for index in order:
            index = int(index)
            if eligible[index] and index not in selected_set:
                selected.append(index)
                selected_set.add(index)
                if len(selected) >= int(top_k):
                    break
    return np.asarray(selected, dtype=np.int64)


def selection_audit(
    selected_indices: Sequence[int],
    source_uids: Sequence[Any],
    event_families: Sequence[Any],
    semantic_labels: Sequence[Any],
) -> Dict[str, Any]:
    selected = np.asarray(selected_indices, dtype=np.int64)
    sources = np.asarray(source_uids, dtype=object)[selected]
    families = np.asarray(event_families, dtype=object)[selected]
    semantics = np.asarray(semantic_labels, dtype=object)[selected]
    return {
        "selected": int(len(selected)),
        "source_coverage": int(len(set(map(str, sources)))),
        "family_coverage": int(len(set(map(str, families)))),
        "semantic_coverage": int(len(set(map(str, semantics)))),
        "source_histogram": dict(Counter(map(str, sources))),
        "family_histogram": dict(Counter(map(str, families))),
        "semantic_histogram": dict(Counter(map(str, semantics))),
    }
