"""Pure independent scalar-auditor tests; no actual cache values are loaded."""
from pathlib import Path
import csv
import gzip
import json
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import audit_q1_stability_diagnostic as audit


class IndependentQ1CacheAuditTests(unittest.TestCase):
    def data(self):
        return np.tile([.04,.039,.05],(6,1)),np.full((6,3),.2),[(.2,.3,.5),(.3,.3,.4),(.6,.3,.1)]

    def test_scalar_complete_epsilon_set_and_weight_diameter(self):
        g,d,w=self.data()
        x,m=audit.scalar_individual(g,d,w,[1,2,3],L=.4,floor=.3,gamma=.1,menu_index=1)
        self.assertEqual(x["feasible_indices"],[0,1]);self.assertEqual(x["near_optimal_indices"],[0,1])
        self.assertAlmostEqual(x["total_variation_diameter"],.1)
        self.assertEqual(len(m),2);self.assertAlmostEqual(x["raw_menu_regret_if_feasible"],.001)

    def test_scalar_mean_uses_fsum_not_ordered_numpy_sum(self):
        g=np.array([[1e16,0.],[1.,0.],[-1e16,0.]])
        x,_=audit.scalar_individual(g,np.zeros_like(g),[(.2,.3,.5),(.3,.3,.4)],[1,2,3],L=.4,floor=.1,gamma=.1,menu_index=0)
        self.assertEqual(x["best_mean_log_growth"],1/3)

    def test_only_training_rows_enter_scalar_counts_or_growth(self):
        g,d,w=self.data()
        a,_=audit.scalar_individual(g,d,w,[1,2,3],L=.4,floor=.3,gamma=.1,menu_index=1)
        g[3:]=np.inf;d[3:]=np.nan
        b,_=audit.scalar_individual(g,d,w,[1,2,3],L=.4,floor=.3,gamma=.1,menu_index=1)
        self.assertEqual(a,b)

    def test_independent_subtraction_boundary_and_count_cap(self):
        g,d,w=self.data();d[0,0]=.4+2e-12
        x,_=audit.scalar_individual(g,d,w,[1,2,3],L=.4,floor=.3,gamma=.1,menu_index=0)
        self.assertEqual(x["event_count_cap"],0);self.assertEqual(x["feasible_indices"],[1])
        self.assertFalse(x["raw_menu_feasible"]);self.assertIsNone(x["raw_menu_regret_if_feasible"])

    def test_null_selected_screen_is_distinct_from_empty(self):
        g,d,w=self.data()
        null,_=audit.scalar_individual(g,d,w,[1,2,3],L=.4,floor=.9,gamma=None,menu_index=None)
        empty,_=audit.scalar_individual(g,d,w,[1,2,3],L=.4,floor=.9,gamma=.1,menu_index=None)
        self.assertIsNone(null["feasible_n"]);self.assertEqual(empty["feasible_n"],0)

    def rows(self,n):
        return [dict(assigned=True,breach=False,log_growth=.03,cagr=float(i),mdd=float(i),volatility=.1,nominal_fees=.2,trade_fraction=1.) for i in range(1,n+1)]

    def test_ordinary_median_explicit_quantile_and_fractional_CVaR(self):
        x=audit.scalar_summary(self.rows(25))
        self.assertEqual(x["median_cagr"],13.);self.assertEqual(x["p10_cagr"],3.)
        self.assertEqual(x["p90_mdd"],23.);self.assertEqual(x["cvar90_mdd"],24.2)
        self.assertEqual(audit.scalar_summary(self.rows(4))["median_mdd"],2.5)

    def test_initial_noninvestment_summary_remains_null(self):
        x=audit.scalar_summary([dict(assigned=False)])
        self.assertEqual(x["financial_n"],0);self.assertEqual(x["coverage"],0.)
        self.assertIsNone(x["breach_fraction"]);self.assertIsNone(x["cvar90_mdd"])

    def test_recursive_checks_reject_shape_null_and_numeric_disagreement(self):
        c=audit.Checks();c.equal(dict(x=[1,.1,None,True]),dict(x=[1,.1,None,True]))
        for e,a in ((None,0),(1,True),(.1,.100001),([1],[1,2]),({"x":1},{"y":1})):
            with self.subTest(expected=e),self.assertRaises(AssertionError):c.equal(e,a)

    def test_csv_checker_preserves_true_null_boolean_and_list_schema(self):
        expected=[dict(i=1,x=None,assigned=False,weights=[.2,.3,.5],g=.03)]
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"table.csv.gz"
            with gzip.open(p,"wt",newline="") as f:
                w=csv.DictWriter(f,fieldnames=expected[0]);w.writeheader();w.writerow(dict(i=1,x="",assigned=False,weights=json.dumps([.2,.3,.5]),g=.03))
            audit.compare_csv(p,expected,audit.Checks())

    def test_development_id_guard(self):
        for ids in ([121],[0],[True],[1,1],[1.]):
            with self.assertRaises(ValueError):audit.validate_ids(ids)

    def test_unfrozen_auditor_cannot_read_cache_numerics(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"plan";f=Path(temp)/"freeze";p.write_text('{}')
            f.write_text(json.dumps(dict(execution_ready=False,plan_sha256=audit.sha(p))))
            with patch.object(audit,"PLAN",p),patch.object(audit,"FREEZE",f),patch.object(audit.np,"load") as loader:
                with self.assertRaises(RuntimeError):audit.authorize()
                loader.assert_not_called()


if __name__=="__main__":unittest.main()
