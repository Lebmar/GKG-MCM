"""Synthetic artifact gates; no original result tables or market files read."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("failure_companion", ROOT / "scripts/build_failure_path_notebook.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def fixture(root):
    for name in ("common_source.py", "common_input.txt", "fan_source.py", "scale_source.py"):
        (root / name).write_text("synthetic local artifact\n")
    sources = {"common_source.py": b.sha(root / "common_source.py")}
    inputs = {"common_input.txt": b.sha(root / "common_input.txt")}
    for folder in (b.MECHANISMS, b.INTERVENTIONS, b.COSTS, b.FANS):
        (root / folder).mkdir(parents=True)
        (root / folder / "synthetic.csv").write_text("not,a,real,result\n")
        outputs = {"synthetic.csv": b.sha(root / folder / "synthetic.csv")}
        if folder in (b.MECHANISMS, b.INTERVENTIONS):
            manifest = dict(status="complete", final_scenarios_loaded=False, sources=sources, sources_at_end=sources,
                            input_sha256=inputs, input_sha256_at_end=inputs, outputs_sha256=outputs)
        elif folder == b.COSTS:
            pp = root / "docs/current_candidate_cost_bridge_protocol.json"
            write(pp, dict(execution_ready=True, input_sha256=inputs, source_sha256=sources))
            manifest = dict(status="passed", final_scenarios_read=False, inputs_and_sources={**inputs, **sources},
                            outputs_sha256=outputs, protocol_sha256=b.sha(pp))
        else:
            fan_source = {"fan_source.py": b.sha(root / "fan_source.py")}
            pp = root / "docs/matched_parameter_fan_contrast_protocol.json"
            write(pp, dict(execution_ready=True, frozen_input_sha256=inputs, execution_sources_sha256=fan_source))
            manifest = dict(status="passed", final_scenarios_read=False, sources_and_inputs_sha256={**inputs, **fan_source},
                            output_sha256=outputs, protocol_sha256=b.sha(pp))
        write(root / folder / "manifest.json", manifest)
        key = "producer_manifest_sha256" if folder == b.FANS else "parent_manifest_sha256"
        write(root / folder / "independent_audit.json", dict(status="passed", **{key: b.sha(root / folder / "manifest.json")},
              sources_and_artifacts_at_start=inputs, sources_and_artifacts_at_end=inputs))
    (root / b.SCALE).mkdir(parents=True)
    (root / b.SCALE / "synthetic.json").write_text("{}\n")
    plan = root / "docs/intervention_scale_equivalence_plan.json"
    scale_sources = {"scale_source.py": b.sha(root / "scale_source.py")}
    write(plan, dict(execution_ready=True, sources_sha256=scale_sources, frozen_input_sha256=inputs))
    closure = {**scale_sources, **inputs, str(plan.relative_to(root)): b.sha(plan)}
    write(root / b.SCALE / "manifest.json", dict(status="complete", plan_sha256=b.sha(plan),
          sources_and_inputs_at_start=closure, sources_and_inputs_at_end=closure,
          outputs_sha256={"synthetic.json": b.sha(root / b.SCALE / "synthetic.json")},
          summary=dict(no_new_market_load=True, new_fits=0, new_policy_simulations=0, admission_changed=False)))


def rebind_audit(root, folder):
    path = root / folder / "independent_audit.json"; a = b.readj(path)
    key = "producer_manifest_sha256" if folder == b.FANS else "parent_manifest_sha256"
    a[key] = b.sha(root / folder / "manifest.json"); write(path, a)


class FailureCompanionGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); fixture(self.root)

    def test_four_bound_audits_and_distinct_scale_validation_pass(self):
        result = b.require_complete(self.root)
        self.assertEqual(len(result["studies"]), 4)
        self.assertIn("scale_manifest", result)

    def test_wrong_current_manifest_audit_binding_is_rejected(self):
        path = self.root / b.MECHANISMS / "independent_audit.json"; a = b.readj(path)
        a["parent_manifest_sha256"] = "0" * 64; write(path, a)
        with self.assertRaisesRegex(RuntimeError, "current-manifest"):
            b.require_complete(self.root)

    def test_explicit_final_values_read_is_rejected(self):
        path = self.root / b.FANS / "manifest.json"; m = b.readj(path)
        m["final_scenarios_read"] = True; write(path, m); rebind_audit(self.root, b.FANS)
        with self.assertRaisesRegex(RuntimeError, "exclude final scenario"):
            b.require_complete(self.root)

    def test_changed_fan_source_bytes_are_rejected(self):
        (self.root / "fan_source.py").write_text("changed after frozen hash\n")
        with self.assertRaisesRegex(RuntimeError, "Frozen source/input/artifact changed"):
            b.require_complete(self.root)

    def test_scale_does_not_claim_a_fifth_independent_audit(self):
        self.assertFalse((self.root / b.SCALE / "independent_audit.json").exists())
        result = b.require_complete(self.root)
        self.assertNotIn(str(b.SCALE), result["studies"])
        self.assertNotIn("audit", result["scale_manifest"])

    def test_scale_changed_source_or_end_snapshot_is_rejected(self):
        path = self.root / b.SCALE / "manifest.json"; m = b.readj(path)
        m["sources_and_inputs_at_end"] = {}; write(path, m)
        with self.assertRaisesRegex(RuntimeError, "saved-array scale diagnosis"):
            b.require_complete(self.root)

    def test_confirmed_figures_cannot_be_overwritten_even_with_refresh(self):
        path = self.root / b.FIGURES / "manifest.json"
        write(path, dict(status="complete", visual_review_confirmed=True))
        original = path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "unconfirmed presentation draft"):
            b.save_figures(self.root, {}, {}, refresh_draft=True)
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
