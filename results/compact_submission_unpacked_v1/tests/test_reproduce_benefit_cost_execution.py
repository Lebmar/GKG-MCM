"""No-market checks for isolated economic-stage replay contract."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('benefit_replay',ROOT/'scripts/reproduce_benefit_cost_execution.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)


class BenefitReplayTests(unittest.TestCase):
    def fixture(self,base,task_count=130):
        base.mkdir()
        (base/'value.json').write_text('{"fee_once":true}\n')
        m=dict(status='complete',final_scenarios_loaded=False,scenario_ids_loaded=list(range(1,121)),training_tasks=task_count,
            primary_method='benefit_cost_one_year',sources={},sources_at_end={},input_sha256={},input_sha256_at_end={},
            completed_utc='t',elapsed_seconds=1,outputs_sha256={'value.json':replay.engine.sha(base/'value.json')})
        (base/'manifest.json').write_text(json.dumps(m))

    def test_exact_copy(self):
        with tempfile.TemporaryDirectory() as td:
            a,b=Path(td)/'a',Path(td)/'b';self.fixture(a);self.fixture(b)
            self.assertEqual(replay.compare_study(a,b)['value.json']['comparison'],'byte_exact')

    def test_rejects_old125_scope(self):
        with tempfile.TemporaryDirectory() as td:
            a,b=Path(td)/'a',Path(td)/'b';self.fixture(a);self.fixture(b,125)
            with self.assertRaises(AssertionError):replay.compare_study(a,b)

    def test_rejects_changed_economic_value(self):
        with tempfile.TemporaryDirectory() as td:
            a,b=Path(td)/'a',Path(td)/'b';self.fixture(a);self.fixture(b)
            (b/'value.json').write_text('{"fee_once":false}\n')
            with self.assertRaises(AssertionError):replay.compare_study(a,b)


if __name__=='__main__':unittest.main()
