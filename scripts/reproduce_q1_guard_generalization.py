"""Isolated same-source replay of the Q1 guard generalization diagnostic.

Copies only the frozen allowlisted sources/inputs into a fresh temporary
workspace, reruns the producer there with the project venv, and compares every
deterministic artifact byte for byte. The NPZ is compared field by field
(array-exact) because ZIP member timestamps are not deterministic; the manifest
is compared after normalizing only the declared runtime fields. This adds no
market observations, validation, selection or parameter adjustment.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'results/q1_guard_generalization_v1'
OUT = ROOT / 'results/q1_guard_generalization_reproduction_v1'
PROTOCOL = Path('docs/q1_guard_generalization_protocol.json')
FREEZE = Path('docs/q1_guard_generalization_source_freeze.json')
PRODUCER = Path('scripts/run_q1_guard_generalization.py')
NORMALIZED_MANIFEST_FIELDS = ('started_utc', 'completed_utc', 'outputs_sha256')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readj(path):
    return json.loads(Path(path).read_text())


def writej(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def npz_fields(path):
    """Return {member_name: raw_member_bytes} independent of ZIP timestamps."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            out[info.filename] = zf.read(info.filename)
    return out


def compare_npz(left, right):
    """Compare two np.savez_compressed archives member by member, array-exact.

    ZIP member timestamps differ between runs, so byte comparison is invalid;
    each member (.npy) is a single named array and is compared by value.
    """
    a, b = npz_fields(left), npz_fields(right)
    if set(a) != set(b):
        raise RuntimeError('NPZ member mismatch')
    fields = 0
    for name in a:
        x = np.load(io.BytesIO(a[name]), allow_pickle=False)
        y = np.load(io.BytesIO(b[name]), allow_pickle=False)
        if x.shape != y.shape or x.dtype != y.dtype:
            raise RuntimeError(f'NPZ array shape/dtype mismatch: {name}')
        if x.dtype.kind in 'bUSO':
            same = np.array_equal(x, y)
        else:
            same = np.allclose(x, y, rtol=0, atol=0, equal_nan=True)
        if not same:
            raise RuntimeError(f'NPZ array mismatch: {name}')
        fields += 1
    return fields


def run(root=ROOT, output=OUT):
    root = Path(root)
    freeze = readj(root / FREEZE)
    manifest = readj(ORIGINAL / 'manifest.json')
    if manifest.get('status') != 'complete' or manifest.get('final_scenarios_loaded') is not False:
        raise RuntimeError('Original complete development-only run required')
    if freeze['protocol_sha256'] != sha(root / PROTOCOL) or manifest['protocol_sha256'] != sha(root / PROTOCOL):
        raise RuntimeError('Protocol identity mismatch')
    dependencies = dict(freeze['source_sha256'])
    dependencies.update(freeze['input_sha256'])
    # The freeze file cannot hash itself; bind it through the producer manifest.
    dependencies[str(FREEZE)] = manifest['source_freeze_sha256']
    for name, digest in dependencies.items():
        if sha(root / name) != digest:
            raise RuntimeError('Frozen dependency changed: ' + name)
    original_outputs = dict(manifest['outputs_sha256'])
    for name, digest in original_outputs.items():
        if sha(ORIGINAL / name) != digest:
            raise RuntimeError('Original output changed: ' + name)

    output = Path(output).resolve()
    if output.exists():
        raise RuntimeError('Refuse overwriting existing replay receipt')
    output.mkdir(parents=True)
    receipt = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                   original_manifest_sha256=sha(ORIGINAL / 'manifest.json'),
                   dependencies_copied=len(dependencies), scope='same_source_isolated_replay')
    writej(output / 'receipt.json', receipt)
    try:
        with tempfile.TemporaryDirectory(prefix='q1_guard_replay_') as tmp:
            workspace = Path(tmp)
            for name in dependencies:
                dest = workspace / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / name, dest)
            env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
            completed = subprocess.run(
                [str(root / '.venv/bin/python'), str(workspace / PRODUCER), '--execute',
                 '--output', str(workspace / 'results/q1_guard_generalization_v1')],
                cwd=workspace, env=env, capture_output=True, text=True)
            (output / 'replay_stdout.txt').write_text(completed.stdout)
            (output / 'replay_stderr.txt').write_text(completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError('Replay producer failed; see replay_stderr.txt')
            replay = workspace / 'results/q1_guard_generalization_v1'
            replay_files = {x.name for x in replay.iterdir()}
            expected = set(original_outputs) | {'manifest.json'}
            if replay_files != expected:
                raise RuntimeError(f'Replay file set mismatch: {sorted(replay_files ^ expected)}')
            compared, npz_arrays = {}, 0
            for name in sorted(original_outputs):
                left, right = ORIGINAL / name, replay / name
                if name.endswith('.npz'):
                    npz_arrays += compare_npz(left, right)
                    compared[name] = 'array_exact'
                else:
                    if left.read_bytes() != right.read_bytes():
                        raise RuntimeError('Byte mismatch: ' + name)
                    compared[name] = 'byte_exact'
            lm, rm = dict(manifest), readj(replay / 'manifest.json')
            for field in NORMALIZED_MANIFEST_FIELDS:
                lm.pop(field, None)
                rm.pop(field, None)
            if lm != rm:
                raise RuntimeError('Manifest mismatch beyond declared runtime fields')
        receipt.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                       compared_outputs=compared, npz_array_fields=npz_arrays,
                       manifest_normalized_fields=list(NORMALIZED_MANIFEST_FIELDS),
                       note='Same-source/same-data computational reproduction; not statistical validation.')
        writej(output / 'receipt.json', receipt)
    except BaseException as exc:
        (output / 'failure.txt').write_text(traceback.format_exc())
        receipt.update(status='failed', error=str(exc),
                       failed_utc=datetime.now(timezone.utc).isoformat())
        writej(output / 'receipt.json', receipt)
        raise
    print(json.dumps(dict(status='complete', output=str(output), npz_array_fields=npz_arrays)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT,
                        help='Fresh receipt directory; a failed attempt is never overwritten.')
    args = parser.parse_args()
    if args.execute:
        run(output=args.output)
    else:
        print(json.dumps(dict(source_only=True, replays=str(ORIGINAL.relative_to(ROOT)),
                              reads_market_values=False), indent=2))
