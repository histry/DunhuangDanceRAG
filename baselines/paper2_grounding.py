"""Offline label-free grounding benchmark on a frozen observable feature bank."""
from __future__ import annotations

import argparse
import time

import numpy as np

from baselines.benchmark_common import array, artifact, load_manifest, sha256, start_output, summary, write_json
from training.weak_semantic_ot import DEFAULT_COST_WEIGHTS, sparse_sinkhorn_teacher, weighted_control_cost


def softmax(scores, temperature):
    z = scores / temperature
    z -= z.max(axis=1, keepdims=True)
    value = np.exp(z)
    return value / value.sum(axis=1, keepdims=True)


def metrics(probabilities, cost, groups, relevance=None, k=10):
    n, e = probabilities.shape
    order = np.argsort(-probabilities, axis=1, kind='stable')
    top = order[:, :min(k,e)]
    mass = probabilities.mean(axis=0)
    group_mass = {g: float(mass[np.asarray(groups)==g].sum()) for g in sorted(set(groups))}
    result = {
        'expected_observable_cost_proxy': float((probabilities*cost).sum(axis=1).mean()),
        'mean_entropy': float(-(probabilities*np.log(np.maximum(probabilities,1e-30))).sum(axis=1).mean()),
        'source_mass': group_mass, 'source_hhi': sum(x*x for x in group_mass.values()),
        'top1_unique_event_ratio': float(len(set(order[:,0]))/n),
        'top_k_event_coverage': float(len(set(top.ravel()))/e),
        'retrieval_metrics': None,
        'proxy_is_independent_quality_evidence': False,
    }
    if relevance is not None:
        valid = relevance.sum(axis=1)>0
        if valid.any():
            recalls, reciprocal, ndcg = [], [], []
            for i in np.flatnonzero(valid):
                rel = relevance[i,order[i]]
                recalls.append(float((rel[:k]>0).sum()/(rel>0).sum()))
                reciprocal.append(1.0/(np.flatnonzero(rel>0)[0]+1))
                weights = 1/np.log2(np.arange(min(k,e))+2)
                ideal = np.sort(relevance[i])[::-1][:k]
                ndcg.append(float((rel[:k]*weights).sum()/(ideal*weights).sum()))
            result['retrieval_metrics'] = {'evaluated_queries': int(valid.sum()), 'k': k,
                'recall_at_k': float(np.mean(recalls)), 'mrr': float(np.mean(reciprocal)),
                'ndcg_at_k': float(np.mean(ndcg)), 'source': 'external_relevance_only'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--methods', nargs='+', default=['cosine', 'control_softmax', 'sinkhorn', 'sparse_ot', 'source_balanced_ot'])
    args = parser.parse_args()
    r, base, digest = load_manifest(args.manifest, 'paper2_grounding_benchmark_v1')
    music = array(r['music_targets'], base)
    motion = array(r['motion_descriptors'], base)
    groups = list(map(str, r['event_source_uids']))
    song_ids = np.asarray(r['query_song_uids'], dtype=str)
    if len(groups)!=len(motion) or len(song_ids)!=len(music):
        raise ValueError('Unaligned query/event identities')
    if len(set(r['event_uids']))!=len(motion) or len(r['query_uids'])!=len(music):
        raise ValueError('Invalid event/query IDs')
    if set(song_ids) & set(r.get('training_song_uids', [])):
        raise ValueError('Evaluation songs overlap training songs')
    cost = weighted_control_cost(music, motion)
    temperature = float(r.get('temperature', .12))
    if not np.isfinite(temperature) or temperature<=0:
        raise ValueError('temperature must be positive')
    k = int(r.get('top_k', 10))
    if not 1<=k<=len(motion):
        raise ValueError('Invalid top_k')
    relevance = None
    if 'relevance' in r:
        if r['relevance'].get('origin') not in ('human_evaluation', 'heldout_paired_dataset'):
            raise ValueError('OT teacher cannot be used as evaluation truth')
        relevance = array(r['relevance'], base)
        if relevance.shape!=cost.shape or np.any(relevance<0):
            raise ValueError('Invalid external relevance')
    out = start_output(args.output_dir, digest, args.methods)
    rows = []
    for index, method in enumerate(args.methods):
        started = time.perf_counter()
        solver_reports = []
        if method == 'cosine':
            w = np.sqrt(DEFAULT_COST_WEIGHTS)
            x, y = music*w, motion*w
            x /= np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
            y /= np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12)
            p = softmax(x@y.T, temperature)
        elif method == 'control_softmax':
            p = softmax(-cost, temperature)
        elif method == 'contrastive_encoder':
            from baselines.frozen_grounding_model import frozen_scores
            entry = r['checkpoints'][method]
            p = softmax(frozen_scores(artifact(entry,base),music,motion,song_ids),temperature)
        elif method in ('sinkhorn', 'sparse_ot', 'source_balanced_ot'):
            p = np.zeros_like(cost)
            # Fit each song independently: no transport through other test songs.
            for song in sorted(set(song_ids)):
                mask = song_ids == song
                balance = groups if method=='source_balanced_ot' else ['uniform']*len(groups)
                probabilities, info = sparse_sinkhorn_teacher(cost[mask], balance,
                    top_k=len(groups) if method=='sinkhorn' else k, epsilon=temperature)
                if not info['converged']:
                    raise RuntimeError(f'OT did not converge for {song}')
                p[mask] = probabilities
                info['song_uid'] = song
                target = np.asarray([1/groups.count(g) for g in groups]) if method=='source_balanced_ot' else np.ones(len(groups))
                active = probabilities.sum(axis=0)>0
                target = target*active
                target /= target.sum()
                info['returned_probability_column_error'] = float(np.abs(probabilities.mean(axis=0)-target).max())
                solver_reports.append(info)
        elif method in r.get('frozen_scores', {}):
            entry = r['frozen_scores'][method]
            artifact(entry['checkpoint'], base)
            if entry['query_uids']!=r['query_uids'] or entry['event_uids']!=r['event_uids']:
                raise ValueError('Frozen score identities differ')
            if set(song_ids)&set(entry['training_song_uids']):
                raise ValueError('Frozen baseline training/evaluation leakage')
            scores = array(entry, base)
            if scores.shape!=cost.shape:
                raise ValueError('Frozen score shape mismatch')
            p = softmax(scores, temperature)
        else:
            raise ValueError(f'Unknown method/missing frozen score: {method}')
        path = out/f'{index:02d}.probabilities.npy'
        np.save(path,p)
        rows.append({'method': method, 'seconds': time.perf_counter()-started,
                     'probability_path': path.name, 'sha256': sha256(path),
                     'metrics': metrics(p,cost,groups,relevance,k), 'solver': solver_reports})
    write_json(out/'identities.json', {key:r[key] for key in ('query_uids','event_uids','query_song_uids','event_source_uids')})
    write_json(out/'report.json', summary(rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
