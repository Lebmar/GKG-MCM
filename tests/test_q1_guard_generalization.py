"""Focused synthetic/unit checks for the Q1 guard generalization producer.

These tests never parse market data. They pin the freeze gate's integrity, the
summary statistic conventions, the paired-comparison partitions, the NULL
handling for unfunded Review arms and the no-overwrite guarantee.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import run_q1_guard_generalization as producer

PROTOCOL = ROOT / 'docs/q1_guard_generalization_protocol.json'
FREEZE = ROOT / 'docs/q1_guard_generalization_source_freeze.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row(object_id='X', scenario_id=1, assigned=True, cagr=0.0, mdd=0.0, breach=False,
        profile='B', horizon=15, L=0.25, **overrides):
    base = dict(object_id=object_id, profile=profile, assignment='Conservative',
                horizon=horizon, L=L, rule='annual', execution_status='funded',
                scenario_id=scenario_id, assigned=assigned,
                log_growth=math.log1p(cagr), cagr=cagr, mdd=mdd, volatility=0.1,
                terminal_real=100.0, nominal_fees=1.0, trade_fraction=0.5,
                breach=breach, first_crossing=None, automatic_fraction=0.9,
                protection_trade_fraction=0.0, support_review=False,
                mitigation_trade_fraction=0.0, mitigation_nominal_fees=0.0)
    base.update(overrides)
    return base


class FreezeGateTest(unittest.TestCase):
    def test_protocol_frozen_and_hashes_current(self):
        freeze = json.loads(FREEZE.read_text())
        self.assertTrue(freeze['execution_ready'])
        self.assertEqual(freeze['protocol_sha256'], sha(PROTOCOL))
        for key in ('source_sha256', 'input_sha256'):
            for name, digest in freeze[key].items():
                self.assertEqual(sha(ROOT / name), digest, name)

    def test_protocol_is_development_only(self):
        p = json.loads(PROTOCOL.read_text())
        self.assertIs(p['allow_final'], False)
        self.assertEqual(p['development_ids'], list(range(1, 121)))
        self.assertEqual(p['locked_guard_spec']['candidate_id'], 'release_0.5_uniform')
        ids = {o['object_id'] for o in p['objects']}
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(p['objects']), 8)

    def test_profiles_own_horizons_and_limits(self):
        p = json.loads(PROTOCOL.read_text())
        b = next(o for o in p['objects'] if o['object_id'] == 'B_guard')
        e = next(o for o in p['objects'] if o['object_id'] == 'E_guard')
        self.assertEqual((b['horizon'], b['L'], b['target']), (15, 0.25, [0.25, 0.30, 0.45]))
        self.assertEqual((e['horizon'], e['L'], e['target']), (6, 0.35, [0.65, 0.05, 0.30]))
        for o in p['objects']:
            self.assertAlmostEqual(sum(o['target']), 1.0, places=12)
            self.assertTrue(all(w > 0 for w in o['target']))

    def test_freeze_gate_passes(self):
        p, f = producer.freeze_gate()
        self.assertEqual(f['protocol_sha256'], sha(PROTOCOL))

    def test_run_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                producer.run(output=Path(tmp))


class SummaryTest(unittest.TestCase):
    def test_summary_conventions_on_synthetic_rows(self):
        rows = [row(scenario_id=i + 1, cagr=v, mdd=m, breach=m > 0.25)
                for i, (v, m) in enumerate(zip([0.01, 0.02, 0.03, 0.04], [0.10, 0.20, 0.30, 0.40]))]
        s = producer.summary(rows)
        self.assertEqual(s['original_n'], 4)
        self.assertEqual(s['financial_n'], 4)
        self.assertEqual(s['breach_count'], 2)
        self.assertAlmostEqual(s['breach_fraction'], 0.5)
        self.assertAlmostEqual(s['median_cagr'], 0.025)
        # Inverse-CDF P10 of four values: index ceil(0.4)-1 = 0.
        self.assertAlmostEqual(s['p10_cagr'], 0.01)
        self.assertAlmostEqual(s['p90_mdd'], 0.40)
        # CVaR90: worst 10% mass of 4 = 0.4 mass -> fractional top atom.
        self.assertAlmostEqual(s['cvar90_mdd'], 0.40)

    def test_summary_null_for_unfunded(self):
        rows = []
        for i in range(3):
            r = row(scenario_id=i + 1, assigned=False)
            for k in producer.FINANCIAL:
                r[k] = None
            rows.append(r)
        s = producer.summary(rows)
        self.assertEqual(s['financial_n'], 0)
        self.assertEqual(s['initial_review_n'], 3)
        self.assertIsNone(s['median_cagr'])
        self.assertIsNone(s['breach_count'])

    def test_summary_rejects_mixed_objects(self):
        rows = [row(object_id='A', scenario_id=1), row(object_id='B', scenario_id=2)]
        with self.assertRaises(ValueError):
            producer.summary(rows)


class CompareTest(unittest.TestCase):
    def test_breach_partition(self):
        model = [row(scenario_id=1, breach=True), row(scenario_id=2, breach=False),
                 row(scenario_id=3, breach=False), row(scenario_id=4, breach=True)]
        ref = [row(scenario_id=1, breach=True), row(scenario_id=2, breach=True),
               row(scenario_id=3, breach=False), row(scenario_id=4, breach=False)]
        comp, paired = producer.compare(model, ref)
        parts = comp['event_id_partition']
        self.assertEqual(parts['both'], [1])
        self.assertEqual(parts['reference_only'], [2])
        self.assertEqual(parts['model_only'], [4])
        self.assertEqual(parts['neither'], [3])
        self.assertEqual(len(paired), 4)

    def test_compare_rejects_id_mismatch(self):
        model = [row(scenario_id=1)]
        ref = [row(scenario_id=2)]
        with self.assertRaises(ValueError):
            producer.compare(model, ref)


class AuditConventionsTest(unittest.TestCase):
    def test_producer_and_audit_agree_on_constants(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        import audit_q1_guard_generalization as audit
        self.assertEqual(producer.TOL, audit.TOL)
        self.assertEqual(producer.DEV_IDS, audit.DEV_IDS)
        self.assertEqual(producer.ROOT, audit.ROOT)


if __name__ == '__main__':
    unittest.main()
