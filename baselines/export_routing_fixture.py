"""Freeze the actual Graph-SB candidate preparation and geometry for route-only comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baselines.benchmark_common import sha256, write_json
from routing.boundary_closed_loop import formal_candidate_state_from_slots
from routing.global_path import _prepare_graph_layers, _build_graph_edges
from support.event_identity import event_uids_from_generation_db


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule',required=True)
    parser.add_argument('--db',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--top-k',type=int,default=20)
    args = parser.parse_args()
    if args.top_k<1:
        raise ValueError('top-k must be positive')
    slots = json.loads(Path(args.schedule).read_text(encoding='utf-8-sig'))['slots']
    with np.load(args.db,allow_pickle=True) as source:
        db = {key:source[key] for key in source.files}
    uids = event_uids_from_generation_db(db)
    _,candidates,_ = formal_candidate_state_from_slots(slots,uids,boundary_top_k=args.top_k)
    layers,targets,traces,policy = _prepare_graph_layers(slots,candidates,db,banned={},topk=args.top_k)
    costs,masks,reports = _build_graph_edges(slots,layers,db)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        path = out/name
        np.save(path,value)
        return {'path': name, 'sha256': sha256(path)}
    manifest = {'schema': 'paper3_routing_benchmark_v1',
        'event_uids': [[str(uids[j]) for j in layer] for layer in layers],
        'probabilities': [save(f'p_{i:04d}.npy',v) for i,v in enumerate(targets)],
        'edge_costs': [save(f'cost_{i:04d}.npy',v) for i,v in enumerate(costs)],
        'feasible_masks': [save(f'mask_{i:04d}.npy',v) for i,v in enumerate(masks)],
        'source_schedule_sha256': sha256(args.schedule), 'source_db_sha256': sha256(args.db),
        'history_policy': policy, 'epsilon': .35, 'beam_size': 32}
    write_json(out/'manifest.json',manifest)
    write_json(out/'geometry_provenance.json',{'traces':traces,'edge_reports':reports})
    return 0


if __name__=='__main__':
    raise SystemExit(main())
