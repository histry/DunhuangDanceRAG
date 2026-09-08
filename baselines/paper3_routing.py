"""Route-only baselines sharing frozen unaries, edge costs and hard masks."""
from __future__ import annotations

import argparse
import time

import numpy as np

from baselines.benchmark_common import array, load_manifest, start_output, summary, write_json
from routing.graph_schrodinger import chain_marginals, multi_marginal_schrodinger, viterbi_path


def choose(method, probabilities, costs, masks, epsilon, beam_size):
    unary = [np.log(np.maximum(p,1e-30)) for p in probabilities]
    # Zero target probability is a hard exclusion for every method.
    unary = [np.where(p>0,u,-np.inf) for p,u in zip(probabilities,unary)]
    edges = [np.where(mask,-c,-np.inf) for c,mask in zip(costs,masks)]
    if method == 'greedy':
        return [int(np.argmax(p)) for p in probabilities], {}
    if method == 'beam':
        states = [(float(v),[j]) for j,v in enumerate(unary[0]) if np.isfinite(v)]
        states = sorted(states,key=lambda x:x[0],reverse=True)[:beam_size]
        for t in range(1,len(unary)):
            candidates = [(s+edges[t-1][path[-1],j]+v,path+[j])
                          for s,path in states for j,v in enumerate(unary[t])
                          if np.isfinite(v+edges[t-1][path[-1],j])]
            states = sorted(candidates,key=lambda x:x[0],reverse=True)[:beam_size]
            if not states:
                raise RuntimeError('Beam exhausted')
        return states[0][1], {'beam_size': beam_size}
    if method in ('viterbi','shortest_path'):
        # On a layered DAG the additive shortest path equals max-sum Viterbi.
        path,_ = viterbi_path(unary[0],edges,[np.zeros_like(unary[0]),*unary[1:]])
        return list(map(int,path)), {'objective': 'negative_log_probability_plus_edge_cost',
                                   'equivalent_algorithm': 'layered_dag_shortest_path'}
    if method == 'entropy_chain':
        initial = unary[0]/epsilon
        transitions = [e/epsilon for e in edges]
        nodes = [np.zeros_like(unary[0]),*[u/epsilon for u in unary[1:]]]
        marginals = chain_marginals(initial,transitions,nodes)
        path,_ = viterbi_path(initial,transitions,nodes)
        return list(map(int,path)), {'node_marginals': marginals.node,
                                   'target_marginals_enforced': False}
    if method == 'graph_sb':
        result = multi_marginal_schrodinger(probabilities,costs,feasible_masks=masks,
                                            epsilon=epsilon)
        if not result.converged:
            raise RuntimeError('Graph-SB IPF did not converge')
        return list(map(int,result.map_path)), {
            'node_marginals': result.node_marginals, 'path_entropy': result.path_entropy,
            'maximum_l1_residual': result.maximum_l1_residual,
            'maximum_fisher_rao_residual': result.maximum_fisher_rao_residual,
            'iterations': result.iterations, 'target_marginals_enforced': True}
    raise ValueError(f'Unknown routing method {method}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--methods', nargs='+', default=['greedy','beam','viterbi','shortest_path','entropy_chain','graph_sb'])
    args = parser.parse_args()
    r,base,digest = load_manifest(args.manifest,'paper3_routing_benchmark_v1')
    layers = r['event_uids']
    probabilities = [array(e,base).astype(float) for e in r['probabilities']]
    costs = [array(e,base).astype(float) for e in r['edge_costs']]
    masks = [array(e,base) for e in r['feasible_masks']]
    if not layers or len(layers)!=len(probabilities) or len(costs)!=len(layers)-1 or len(masks)!=len(costs):
        raise ValueError('Layer counts mismatch')
    for ids,p in zip(layers,probabilities):
        if p.shape!=(len(ids),) or len(set(ids))!=len(ids) or not len(ids) or np.any(p<0) or not np.isclose(p.sum(),1):
            raise ValueError('Malformed frozen candidate layer')
    for t,(c,mask) in enumerate(zip(costs,masks)):
        if c.shape!=(len(layers[t]),len(layers[t+1])) or mask.shape!=c.shape or not np.isin(mask,[0,1]).all():
            raise ValueError('Edge shape or mask mismatch')
    masks = [x.astype(bool) for x in masks]
    epsilon = float(r.get('epsilon',.35))
    beam_size = int(r.get('beam_size',32))
    if not np.isfinite(epsilon) or epsilon<=0 or beam_size<1:
        raise ValueError('Invalid solver parameters')
    out = start_output(args.output_dir,digest,args.methods)
    rows = []
    for method in args.methods:
        started = time.perf_counter()
        try:
            path,detail = choose(method,probabilities,costs,masks,epsilon,beam_size)
            if len(path)!=len(layers) or any(j<0 or j>=len(layers[t]) for t,j in enumerate(path)):
                raise RuntimeError('Invalid route')
            bad = [t for t in range(len(costs)) if not masks[t][path[t],path[t+1]]]
            ids = [layers[t][j] for t,j in enumerate(path)]
            energy = sum(-np.log(max(probabilities[t][j],1e-30)) for t,j in enumerate(path))
            edge_values = [float(c[path[t],path[t+1]]) for t,c in enumerate(costs)]
            rows.append({'method': method, 'status': 'INFEASIBLE' if bad else 'OK',
                         'violating_edges': bad, 'chosen_event_uids': ids, 'local_path': path,
                         'unary_nll': float(energy), 'edge_cost_sum': sum(edge_values),
                         'maximum_edge_cost': max(edge_values,default=0),
                         'unique_event_ratio': len(set(ids))/len(ids),
                         'seconds': time.perf_counter()-started, 'solver': detail})
        except RuntimeError as exc:
            rows.append({'method': method, 'status': 'INFEASIBLE', 'reason': str(exc),
                         'seconds': time.perf_counter()-started})
    report = summary(rows)
    report['formal_generation_acceptance'] = False
    report['scope'] = 'frozen_layered_graph_only; history constraints must be encoded in states/masks'
    write_json(out/'report.json',report)
    return 0 if all(row['status']=='OK' for row in rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())
