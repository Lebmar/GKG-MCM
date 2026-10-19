"""Independent selector arithmetic checks; no project selector or market I/O."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_joint_liquidity_refit as audit


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


if __name__ == "__main__":
    unittest.main()
