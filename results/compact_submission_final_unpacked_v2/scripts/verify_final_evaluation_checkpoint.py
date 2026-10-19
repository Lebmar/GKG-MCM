"""Archive final verification while preserving the earlier development checkpoint.

Only saved JSON metadata, byte hashes, the completed test log and Git state
are inspected. No market parser, policy, forecast evaluator or fitter runs.
"""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import sys

from verify_research_checkpoint import HASH_KEYS, PAIRS, evidence_records

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT/'results/final_locked_evaluation_v1'
REPLAY = ROOT/'results/final_locked_reproduction_v1'
ANALYSIS = ROOT/'results/final_locked_analysis_v1'
FIGURES = ROOT/'research/figures/locked_final_evaluation'
OUT = ROOT/'results/final_locked_checkpoint_v1'
EXTRA_HASH_KEYS = {'analysis_source_sha256', 'reproducer_sources', 'frozen_copy_allowlist_without_final_release_sha256'}
OUTPUTS = {'Q1_paths.csv.gz', 'Q2_paths.csv.gz', 'summary.csv.gz', 'evaluation_traces.npz',
           'object_specification.json', 'existing_development_context.json'}


def readj(path):
    return json.loads(Path(path).read_text())


def raw_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise RuntimeError('Do not overwrite the final verification checkpoint')
    actual_hashes = {}; checks = 0

    def sha(path):
        path = Path(path).resolve()
        if path not in actual_hashes:
            actual_hashes[path] = raw_sha(path)
        return actual_hashes[path]

    def mapping(values, prefix):
        nonlocal checks
        for name, digest in values.items():
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                raise AssertionError('Complete SHA256 map required')
            assert sha(prefix/name) == digest, (prefix, name)
            checks += 1

    # Keep the entire previously verified development evidence and its897-test
    # log intact; the new final test log lives in a different archive.
    development = ROOT/'results/model_review/research_checkpoint_v1/verification.json'
    old = readj(development)
    assert old['status'] == 'passed' and old['unit_tests']['passed'] == 897
    assert old['saved_snapshots_checked'] == len(old['snapshots_sha256']) == 189
    assert sha(ROOT/old['unit_tests']['log_path']) == old['unit_tests']['log_sha256']
    mapping(old['snapshots_sha256'], ROOT)
    paths = [ROOT/name for name in old['snapshots_sha256']]
    paths += list((ROOT/'docs').glob('final_evaluation_*.json'))
    paths += list((ROOT/'docs').glob('final_locked_*.json'))
    paths += [FINAL/'manifest.json', FINAL/'independent_audit.json', REPLAY/'manifest.json',
              ANALYSIS/'manifest.json', FIGURES/'manifest.json', development,
              ROOT/'docs/locked_final_evaluation_interpretation_judgment.json',
              ROOT/'results/final_locked_audit_attempt_v1/attempt.json',
              FIGURES/'draft_manifest.json',
              ROOT/'scripts/verify_research_checkpoint.py', Path(__file__)]
    for path in sorted(set(paths)):
        if path.suffix != '.json':
            sha(path); continue
        item = readj(path); sha(path)
        for record in evidence_records(item):
            for before, after in PAIRS:
                if before in record and after in record:
                    assert record[before] == record[after], (path, before, after)
                    checks += 1
            for key in HASH_KEYS | EXTRA_HASH_KEYS:
                values = record.get(key)
                if not isinstance(values, dict):
                    continue
                for name, expected in values.items():
                    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
                        continue
                    candidates = [ROOT/name, path.parent/name]
                    if key.startswith('output'):
                        candidates.reverse()
                    candidate = next((p for p in candidates if p.is_file()), None)
                    assert candidate is not None and sha(candidate) == expected, (path, key, name)
                    checks += 1

    release_path = ROOT/'docs/final_evaluation_release.json'
    prep_path = ROOT/'docs/final_evaluation_protocol.json'
    release, prep, m = readj(release_path), readj(prep_path), readj(FINAL/'manifest.json')
    assert release['final_execution_ready'] is True and prep['final_execution_ready'] is False
    old_git = subprocess.run(['git','show',release['predecessor_checkpoint_commit']+':results/model_review/research_checkpoint_v1/verification.json'],
                             cwd=ROOT,capture_output=True,check=True)
    assert hashlib.sha256(old_git.stdout).hexdigest() == sha(development)
    assert release['preparation_protocol_sha256'] == sha(prep_path)
    assert release['execution_sources_sha256'] == prep['execution_sources_sha256']
    assert release['frozen_input_sha256'] == prep['frozen_input_sha256']
    assert release['object_specification_sha256'] == prep['object_specification_sha256']
    assert release['final_audit_plan_sha256'] == sha(ROOT/'docs/final_locked_evaluation_audit_plan.json')
    assert release['reproduction_plan_sha256'] == sha(ROOT/'docs/final_evaluation_reproduction_plan.json')
    assert release['analysis_plan_sha256'] == sha(ROOT/'docs/final_evaluation_analysis_plan.json')
    assert m['status'] == 'complete' and m['release_sha256'] == sha(release_path)
    assert m['preparation_protocol_sha256'] == sha(prep_path)
    assert m['object_specification_sha256'] == prep['object_specification_sha256']
    assert m['training_ids_loaded'] == list(range(1,121)) and m['evaluation_ids_loaded'] == list(range(121,181))
    assert m['sources'] == m['sources_at_end'] == prep['execution_sources_sha256']
    assert m['input_sha256'] == m['input_sha256_at_end'] == prep['frozen_input_sha256']
    assert (m['Q2_path_rows'],m['Q1_path_rows'],m['summary_rows']) == (240,300,9)
    assert m['new_fits'] == 0 and m['selection_changed'] is False and m['admission_changed'] is False
    assert m['current_C_service'] == 'execution_review' and m['current_C_service_financials'] is None
    assert m['final_scenarios_loaded'] is True and m['completed_final_array_load'] is True
    assert set(m['outputs_sha256']) == OUTPUTS
    mapping(m['outputs_sha256'], FINAL)
    assert sha(ROOT/'docs/main_model_registry.json') == '2d73f73d47523c922329c59c78548f8bc67a8bb93d4e8d3eef67a5bd1d2e7074'

    audit = readj(FINAL/'independent_audit.json')
    assert audit['status'] == 'passed' and audit['parent_manifest_sha256'] == sha(FINAL/'manifest.json')
    assert audit['release_sha256'] == sha(release_path)
    assert audit['counts']['accounts'] == 360 and audit['counts']['periods'] == 8460
    assert audit['counts']['review_rows'] == 180 and len(audit['fixed_Q_witnesses']) == 9
    assert audit['audit_plan_sha256'] == sha(ROOT/'docs/final_locked_evaluation_audit_plan.json')
    assert audit['audit_source_freeze_sha256'] == sha(ROOT/'docs/final_locked_evaluation_audit_source_freeze.json')
    attempt = readj(ROOT/'results/final_locked_audit_attempt_v1/attempt.json')
    assert attempt['status'] == 'passed' and attempt['receipt_sha256'] == sha(FINAL/'independent_audit.json')
    assert attempt['parent_manifest_sha256'] == sha(FINAL/'manifest.json')
    assert attempt['auditor_source_sha256'] == sha(ROOT/'scripts/audit_final_locked_evaluation.py')
    replay = readj(REPLAY/'manifest.json'); comp = readj(REPLAY/'comparison.json')
    assert replay['status'] == 'passed' and replay['differences'] == 0 and comp['differences'] == []
    assert replay['copied_sources_and_inputs_unchanged'] and replay['original_producer_archive_unchanged']
    assert replay['plan_sha256'] == sha(ROOT/'docs/final_evaluation_reproduction_plan.json')
    assert replay['source_freeze_sha256'] == sha(ROOT/'docs/final_evaluation_reproduction_source_freeze.json')
    assert replay['original_final_producer_sha256_start'] == replay['original_final_producer_sha256_end']
    mapping(replay['original_final_producer_sha256_start'], FINAL)
    assert comp['manifest_equal_after_declared_normalization'] and comp['normalized_manifest_keys'] == ['completed_utc']
    assert comp['npz']['exact'] and comp['npz']['fields_checked'] == 234 and comp['npz']['container_bytes_equal']
    assert len(comp['files']) == 6 and all(x['byte_equal'] for x in comp['files'].values())
    allowlist = readj(REPLAY/'copy_allowlist.json')
    assert allowlist == readj(REPLAY/'copied_dependencies_at_end.json') and len(allowlist) == 46
    mapping(allowlist, ROOT)
    interpretation = readj(ANALYSIS/'manifest.json')
    assert interpretation['status'] == 'complete' and interpretation['parent_manifest_sha256'] == sha(FINAL/'manifest.json')
    assert interpretation['analysis_plan_sha256'] == release['analysis_plan_sha256']
    assert interpretation['paired_rows'] == 120 and interpretation['paired_comparisons'] == 2
    assert interpretation['model_or_admission_changed'] is False
    assert sha(ANALYSIS/'official_summary.csv.gz') == sha(FINAL/'summary.csv.gz')
    judgment = readj(ROOT/'docs/locked_final_evaluation_interpretation_judgment.json')
    verified = judgment['numerical_verification']
    assert verified['producer_manifest_sha256'] == sha(FINAL/'manifest.json')
    assert verified['independent_audit']['receipt_sha256'] == sha(FINAL/'independent_audit.json')
    assert verified['isolated_same_source_reproduction']['manifest_sha256'] == sha(REPLAY/'manifest.json')
    assert verified['analysis_manifest_sha256'] == sha(ANALYSIS/'manifest.json')
    assert verified['independent_paired_arithmetic']['checks'] == 1030
    assert verified['independent_paired_arithmetic']['maximum_absolute_difference'] == 0
    assert judgment['primary_service']['auto_n'] == 0 and judgment['primary_service']['financial_outputs'] is None
    assert judgment['changes_in_this_review']['producer_or_model_modified'] is False
    figure = readj(FIGURES/'manifest.json')
    assert figure['status'] == 'complete' and figure['author_visual_QA'] is True and figure['root_visual_QA'] is True
    assert figure['draft_manifest_sha256'] == sha(FIGURES/'draft_manifest.json')
    assert figure['paired_n'] == 60 and figure['original_ids'] == list(range(121,181))
    assert figure['policy_or_admission_changed'] is False and figure['main_C_finance'] is None

    testlog = ROOT/'tmp/final_evaluation_tests.log'
    tests = testlog.read_text(); match = re.search(r'Ran (\d+) tests in ([\d.]+)s', tests)
    assert match and tests.rstrip().endswith('OK') and int(match.group(1)) >= 942
    pinned = {}
    for line in (ROOT/'requirements-research-pinned.txt').read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            name, expected = line.split('=='); assert version(name) == expected; pinned[name] = expected
    assert sys.version_info[:2] == (3,12)
    legacy = subprocess.run(['git','diff','--exit-code','7bca10b','--','src','report','output'],cwd=ROOT,capture_output=True)
    assert legacy.returncode == 0
    sha(testlog)
    # Fresh reads close the time window; do not reuse cached digests here.
    for path, expected in actual_hashes.items():
        assert raw_sha(path) == expected, ('changed during checkpoint',path)
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'unit_tests.txt').write_bytes(testlog.read_bytes())
    result = dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Byte/lineage/complete audit and replay checks; no market parser, model, forecast or new fit.',
        unchanged_development_checkpoint_sha256=sha(development), distinct_hashed_files=len(actual_hashes), hash_and_map_checks=checks,
        checked_current_file_sha256={str(p.relative_to(ROOT)):d for p,d in sorted(actual_hashes.items())},
        final_manifest_sha256=sha(FINAL/'manifest.json'), independent_audit_sha256=sha(FINAL/'independent_audit.json'),
        reproduction_manifest_sha256=sha(REPLAY/'manifest.json'), analysis_manifest_sha256=sha(ANALYSIS/'manifest.json'),
        unit_tests=dict(passed=int(match.group(1)),seconds=float(match.group(2)),log_path='results/final_locked_checkpoint_v1/unit_tests.txt',log_sha256=sha(testlog)),
        runtime=dict(python=sys.version.split()[0],direct_dependencies=pinned),
        original_final_IDs=list(range(121,181)),new_fits=0,admission_changed=False,current_C='execution_review',current_C_financials=None,
        prior_exposure='OldV1 exposure disclosed; locked final reuse and same-input replay are not a freshblind experiment or new market samples.',
        legacy=dict(baseline_revision='7bca10b',unchanged_paths=['src','report','output']))
    (OUT/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed',hashed_files=len(actual_hashes),checks=checks,tests=int(match.group(1)))))


if __name__ == '__main__':
    main()
