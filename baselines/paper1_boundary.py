"""Fixed-window boundary benchmark using the existing exact acceptance audits."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import time

import numpy as np

from baselines.benchmark_common import array, artifact, load_manifest, sha256, start_output, summary, write_json


def audit(candidate, reference, seam, cfg, span):
    from training import motion_models as m
    before = m._safe_validation_audit(reference, cfg, role='benchmark_reference', support_policy='source_observation')
    after = m._safe_validation_audit(candidate, cfg, role='benchmark_candidate', support_policy='source_observation')
    physical = m.evaluate_stage_candidate(before, after, require_repair_gain=False,
                                         ignored_layers=('long_horizon_root_drift',))
    absolute = m.evaluate_physical_audit(after)
    fidelity = m.evaluate_stage_reference_fidelity(before, after)
    support = m._fixed_support_stage_gate(reference, candidate, cfg, before_audit=before, after_audit=after)
    observable = m._observable_boundary_audit(candidate, reference, seam, cfg)
    a, b = span
    outside = np.array_equal(candidate[:a], reference[:a]) and np.array_equal(candidate[b:], reference[b:])
    contact = np.array_equal(candidate[:, :4], reference[:, :4])
    passed = all((physical.get('accepted', False), absolute.get('ok', False),
                  fidelity.get('accepted', False), support.get('accepted', False),
                  observable.get('accepted', False), outside, contact))
    return {'joint_pass': bool(passed), 'physical': physical, 'absolute_physical': absolute,
            'fixed_support': support, 'fidelity': fidelity, 'observable': observable,
            'ownership_unchanged': outside, 'contact_channels_unchanged': contact,
            'reference_metrics': before, 'candidate_metrics': after}


def generate(method, row, reference, seam, cfg, base):
    from training import motion_models as m
    from motion_geometry.heading import resample_motion_so3
    a, b = map(int, row['edit_span'])
    out = reference.copy()
    if method == 'reference':
        return out
    if method in row.get('frozen_outputs', {}):
        entry = row['frozen_outputs'][method]
        artifact(entry['checkpoint'], base)
        if entry['reference_sha256'] != row['reference']['sha256']:
            raise ValueError('Frozen output belongs to a different input')
        return array(entry, base).astype(np.float32)
    if method == 'linear_rot6d':
        t = np.linspace(0, 1, b-a+2)[1:-1, None]
        out[a:b, 4:] = (1-t)*reference[a-1, 4:] + t*reference[b, 4:]
        return out
    if method == 'slerp':
        out[a:b, 4:] = resample_motion_so3(reference[[a-1, b]],
                                         np.linspace(0, 1, b-a+2)[1:-1])[:, 4:]
        return out
    if method == 'context_transformer':
        from baselines.frozen_boundary_model import infer
        bridge = generate('so3_bridge', row, reference, seam, cfg, base)
        return infer(artifact(row['checkpoints'][method],base), bridge, (a,b),
                     row['recording_uid'])
    if method in ('so3_bridge', 'so3_bridge_ik'):
        out[a:b, 4:] = m.reference_motion_inbetween_np(
            reference[max(0,a-4):a], reference[b:b+4], b-a, cfg,
            contact_ik=False)[:, 4:]
        if method.endswith('_ik'):
            out, _ = m.true_lower_body_ik(out, cfg, repair_windows=[(a,b)],
                sliding_support_eligible=array(row['eligible'], base))
        return out
    if method in ('refiner', 'diffusion'):
        checkpoint = artifact(row['checkpoints'][method], base)
        condition = array(row['condition'], base)
        eligible = array(row['eligible'], base)
        fn = m.apply_refiner_model if method == 'refiner' else m.apply_diffusion_model
        return fn(out, condition, seam, str(checkpoint), cfg, sliding_support_eligible=eligible)
    raise ValueError(f'Unknown method or missing frozen output: {method}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--methods', nargs='+', default=['reference', 'linear_rot6d', 'slerp', 'so3_bridge', 'so3_bridge_ik'])
    args = parser.parse_args()
    manifest, base, digest = load_manifest(args.manifest, 'paper1_boundary_benchmark_v1')
    from training import motion_models as m
    cfg = m.MotionGenerationConfig.from_json(str(artifact(manifest['config'], base)))
    for key in ('checkpoint_validation_min_geometry_repair_gain',
                'checkpoint_validation_min_temporal_repair_gain',
                'checkpoint_validation_min_endpoint_repair_gain'):
        if not np.isclose(float(getattr(cfg, key)), .03, rtol=0, atol=1e-12):
            raise ValueError(f'Benchmark requires unchanged 0.03 gate: {key}')
    out = start_output(args.output_dir, digest, args.methods)
    if not manifest.get('cases'):
        raise ValueError('No cases')
    rows = []
    with gzip.open(out / 'audits.jsonl.gz', 'wt', encoding='utf-8') as full:
        for index, case in enumerate(manifest['cases']):
            reference = array(case['reference'], base).astype(np.float32)
            seam = array(case['seam'], base)
            a, b = map(int, case['edit_span'])
            if reference.ndim != 2 or reference.shape[1] != 151 or not 1 <= a < b < len(reference):
                raise ValueError('Invalid EDGE151 reference/edit_span')
            for method_index, method in enumerate(args.methods):
                seed = int(manifest.get('seed', 42)) + index
                np.random.seed(seed)
                if m.torch is not None:
                    m.torch.manual_seed(seed)
                started = time.perf_counter()
                candidate = generate(method, case, reference.copy(), seam.copy(), copy.deepcopy(cfg), base)
                if candidate.shape != reference.shape or not np.isfinite(candidate).all():
                    raise ValueError('Invalid candidate; no fallback')
                result = audit(candidate, reference, seam, cfg, (a,b))
                name = f'{index:05d}_{method_index:02d}.npy'
                np.save(out / name, candidate)
                row = {'case_id': case['case_id'], 'method': method, 'seed': seed,
                       'joint_pass': result['joint_pass'], 'motion': name,
                       'motion_sha256': sha256(out/name), 'seconds': time.perf_counter()-started}
                full.write(json.dumps({**row, 'audit': result}, default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
                rows.append(row)
    report = summary(rows)
    report['by_method'] = {
        method: {'cases': len([r for r in rows if r['method']==method]),
                 'joint_pass_rate': float(np.mean([r['joint_pass'] for r in rows if r['method']==method]))}
        for method in args.methods}
    write_json(out/'report.json', report)
    return 0 if all(r['joint_pass'] for r in rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())
