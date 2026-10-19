"""Independent selector arithmetic checks; no project selector or market I/O."""
from pathlib import Path
from copy import deepcopy
import gzip
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_refit_stability as audit


def evidence():
    return [dict(training_beta=b, eligible=b >= .05, case_auto_n=96,
                 conditional_mean_log_growth=.03, breach_count=6)
            for b in audit.BETAS]


class IndependentSelectorAuditTests(unittest.TestCase):
    def test_refit_checks_only_top_set_and_stops_at_first_feasible_budget(self):
        beta, order, attempts, tied = audit.independent_refit_aware(
            evidence(), [(6, .04)] + [(7, .03)] * 8, 9, 96)
        self.assertEqual(tied, [.05, .075, .1])
        self.assertEqual((beta, order), (.075, 0))
        self.assertEqual([a["training_breach_cap"] for a in attempts], [4, 7])
        self.assertEqual([a["training_beta"] for a in attempts], [.05, .075])

    def test_refit_feasible_budget_outside_inner_top_set_cannot_rescue(self):
        rows = evidence()
        for row in rows:
            if row["training_beta"] > .05:
                row["conditional_mean_log_growth"] = .02
        beta, order, attempts, tied = audit.independent_refit_aware(
            rows, [(6, .04)] + [(7, .03)] * 8, 9, 96)
        self.assertEqual(tied, [.05])
        self.assertIsNone(beta)
        self.assertIsNone(order)
        self.assertEqual(len(attempts), 1)

    def test_one_global_growth_band_cannot_accumulate_pairwise_tolerance(self):
        rows = evidence()
        for row, value in zip(rows[2:], [1., 1. - .75 * audit.TOL,
                                        1. - 1.5 * audit.TOL], strict=True):
            row["conditional_mean_log_growth"] = value
        eligible, tied, coverage, maximum = audit.independent_top_set(rows)
        self.assertEqual(eligible, [.05, .075, .1])
        self.assertEqual(tied, [.05, .075])
        self.assertEqual((coverage, maximum), (96, 1.))

    def test_no_eligible_inner_evidence_never_defaults_to_a_training_winner(self):
        rows = [{**r, "eligible": False} for r in evidence()]
        self.assertEqual(audit.independent_refit_aware(
            rows, [(0, .1)] * 9, 0, 96), (None, None, [], []))


def repeated_protocol():
    """Synthetic identity plan only; no market/source protocol read."""
    contexts, jobs, static = [], [], {}
    original = list(range(1, 121)); base = 87020261018000
    for repeat in range(1, 6):
        chunks = np.array_split(np.random.default_rng(base + repeat).permutation(original), 5)
        for fold, chunk in enumerate(chunks, 1):
            cid = 100 * repeat + fold
            train = sorted(set(original) - set(chunk))
            seed = base + 1000 + cid
            inner = []
            for i, selected in enumerate(np.array_split(np.random.default_rng(seed).permutation(train), 4), 1):
                inner.append(dict(inner_fold=i, test_ids=sorted(map(int, selected)),
                                  train_ids=sorted(set(train) - set(selected))))
                jobs.append(dict(job_id=f"c{cid}_inner{i}"))
            context = dict(context_id=cid, name=f"stability_r{repeat}_outer{fold}",
                           training_ids=train, outer_test_ids=sorted(map(int, chunk)),
                           inner_seed=seed, inner_folds=inner, repeat_id=repeat, outer_fold=fold)
            contexts.append(context); jobs.append(dict(job_id=f"c{cid}_refit"))
            static[f"c{cid}_refit"] = deepcopy(context)
    return dict(contexts=contexts, jobs=jobs, static_contexts=static)


class RepeatedIdentityAuditTests(unittest.TestCase):
    def test_reconstructs25contexts125jobs_without_reading_market(self):
        self.assertEqual(audit.validate_repeated_contexts(repeated_protocol()), 25)

    def test_rejects_same_scenario_test_twice_within_a_repeat(self):
        protocol = repeated_protocol()
        protocol["contexts"][1]["outer_test_ids"] = protocol["contexts"][0]["outer_test_ids"]
        with self.assertRaisesRegex(AssertionError, "Outer seeded IDs"):
            audit.validate_repeated_contexts(protocol)

    def test_rejects_separate_final_static_partition(self):
        protocol = repeated_protocol()
        protocol["static_contexts"]["c101_refit"]["inner_seed"] += 1
        with self.assertRaisesRegex(AssertionError, "partition coupling"):
            audit.validate_repeated_contexts(protocol)

    def test_rejects_missing_training_job_even_when_contexts_complete(self):
        protocol = repeated_protocol(); protocol["jobs"].pop(7)
        with self.assertRaisesRegex(AssertionError, "125-job"):
            audit.validate_repeated_contexts(protocol)

    def test_rejects_inconsistent_repeated_row_annotation(self):
        contexts = repeated_protocol()["contexts"]
        rows = pd.DataFrame([dict(context="stability_r1_outer1",repeat_id=1,outer_fold=1),
                             dict(context="stability_r2_outer3",repeat_id=2,outer_fold=3)])
        self.assertEqual(audit.validate_repeated_table_identities(rows, contexts), 2)
        rows.loc[1,"repeat_id"] = 1
        with self.assertRaisesRegex(AssertionError, "Mislabelled"):
            audit.validate_repeated_table_identities(rows, contexts)

    def test_plain_and_gzip_evidence_have_same_json_identity(self):
        item={"context_id":101,"scenario_ids":[1,2,3],"financial_outcome":None}
        with tempfile.TemporaryDirectory() as folder:
            plain=Path(folder)/"evidence.json"; packed=Path(folder)/"evidence.json.gz"
            plain.write_text(json.dumps(item),encoding="utf-8")
            with gzip.open(packed,"wt",encoding="utf-8") as output:json.dump(item,output)
            self.assertEqual(audit.readj(plain),item)
            self.assertEqual(audit.readj(packed),item)


if __name__ == "__main__":
    unittest.main()
