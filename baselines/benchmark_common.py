"""Frozen-input, inference-only benchmark IO. No training or production writes."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def artifact(entry, base):
    path = Path(entry['path'])
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if sha256(path) != entry['sha256'].lower():
        raise ValueError(f'Artifact SHA256 mismatch: {path}')
    return path


def array(entry, base):
    value = np.load(artifact(entry, base), allow_pickle=False)
    if not isinstance(value, np.ndarray) or not np.isfinite(value).all():
        raise ValueError('Expected a finite NPY array')
    return value


def load_manifest(path, schema):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8-sig'))
    if manifest.get('schema') != schema:
        raise ValueError(f'Expected schema {schema}')
    return manifest, path.parent, sha256(path)


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     default=convert, allow_nan=False), encoding='utf-8')


def start_output(path, manifest_hash, method_names):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=repo)
    sources = {str(p.relative_to(repo)): sha256(p) for p in (repo / 'baselines').glob('*.py')}
    write_json(path / 'provenance.json', {
        'manifest_sha256': manifest_hash, 'commit': commit,
        'tracked_diff_sha256': hashlib.sha256(diff).hexdigest(),
        'runner_source_sha256': sources, 'python': platform.python_version(),
        'methods': method_names, 'training_started': False,
        'production_model_modified': False, 'completed': False,
    })
    return path


def summary(rows):
    return {'completed': True, 'training_started': False,
            'production_model_modified': False, 'results': rows}
