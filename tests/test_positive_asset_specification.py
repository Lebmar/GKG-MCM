"""Index-space and fitting-boundary tests; no market data are loaded."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "research"), str(ROOT / "src")]
import run_positive_asset_specification as strict
import run_nested_static_calibration as nested
from safeinvest_research.static import simplex_grid


def test_exact_subset_and_scenario_axis_mapping():
    weights = simplex_grid(.05)
    values = np.arange(120 * len(weights), dtype=float).reshape(120, len(weights))
    allowed, small, fields = strict.subset_family(weights, {5: {"mdd": values}})
    assert len(allowed) == 172 and len(weights) == 232
    assert np.all(small >= .05 - 1e-12) and np.all(np.diff(allowed) > 0)
    assert np.sum(np.all(np.isclose(small, 1 / 3, atol=1e-15, rtol=0), axis=1)) == 1
    assert np.array_equal(fields[5]["mdd"], values[:, allowed])
    assert np.array_equal(small, weights[allowed])
    assert not np.array_equal(fields[5]["mdd"][:, 0], values[:, 0])


def test_annotations_cover_all_candidate_reference_forms_without_mutation():
    allowed = np.array([4, 17, 21])
    source = {"candidate_index": 1, "selection": {"identified_indices": [0, 2], "optima_by_size": [{"indices": [1]}],
              "search_pool_indices": [0, 1, 2]}, "completion": {"indices": [1, 2], "extension_indices": [2], "reference_frontier": [0, 2]},
              "review": {"candidate_index": None}}
    before = deepcopy(source); result = strict.annotate_indices(source, allowed)
    assert source == before and result["candidate_index"] == 1
    assert result["candidate_original232_index"] == 17
    assert result["selection"]["identified_indices_original232"] == [4, 21]
    assert result["selection"]["optima_by_size"][0]["indices_original232"] == [17]
    assert result["completion"]["reference_frontier_original232"] == [4, 21]
    assert result["review"]["candidate_original232_index"] is None
    with unittest.TestCase().assertRaises(ValueError): strict.annotate_indices({"candidate_index": 3}, allowed)


def test_path_lookup_maps_local_to_global_and_review_nulls():
    fields = {key: np.arange(12, dtype=float).reshape(3, 4) + j for j, key in enumerate(["log_growth", "cagr", "mdd", "volatility", "terminal_real"])}
    row = {"assigned": True, "candidate_index": 0, "scenario_id": 2, "horizon": 5,
           **{key: float(value[1, 2]) for key, value in fields.items()}}
    review = {"assigned": False, **{key: None for key in ["candidate_index", "mdd", "cagr", "log_growth", "breach", "excess"]}}
    strict.verify_path_records([row, review], {5: fields}, np.array([2, 3]))
    with unittest.TestCase().assertRaises(RuntimeError): strict.verify_path_records([row], {5: fields}, np.array([0, 1]))


def test_outer_values_never_enter_calibrate_context_fits():
    protocol = json.loads((ROOT / "docs/positive_asset_specification_protocol.json").read_text())
    context = protocol["contexts"][1]
    train = np.asarray(context["training_ids"]) - 1
    outer = np.asarray(context["outer_test_ids"]) - 1
    weights = np.array([[.2, .3, .5], [.4, .3, .3], [.6, .2, .2]])
    rng = np.random.default_rng(873)
    metrics = {h: {key: rng.uniform(.01, .09, (120, 3)) for key in ["log_growth", "cagr", "mdd", "volatility", "terminal_real"]} for h in [5, 6, 15, 25, 30]}
    changed = deepcopy(metrics)
    for fields in changed.values():
        for values in fields.values(): values[outer] = np.nan
    calls = []
    def fake_fit(metrics, weights, profiles, indices, gamma):
        indices = np.asarray(indices)
        assert set(indices).issubset(set(train)) and not set(indices) & set(outer)
        assert all(np.isfinite(values[indices]).all() for fields in metrics.values() for values in fields.values())
        calls.append(tuple(indices))
        return {"menu": [], "assignments": {p["name"]: "Review" for p in profiles}, "client_product_details": [], "solver_gamma": gamma}
    with patch.object(nested, "fit_gamma", fake_fit):
        original = nested.calibrate_context(metrics, weights, protocol["clients"], context)
        mutated = nested.calibrate_context(changed, weights, protocol["clients"], context)
    assert original == mutated
    assert len(calls) == 40  # 20 inner fits per call; zero coverage gives no context refit.


def test_fit_positive_weights_and_mapped_detail_integrity():
    weights = np.array([[.2, .3, .5], [.4, .3, .3]])
    global_weights = np.vstack(([0., 0., 1.], weights))
    fitted = {"menu": [{"name": "Balanced", "candidate_index": 1, "weights": weights[1].tolist()}],
              "assignments": {"C": "Balanced"}, "client_product_details": [{"portfolio": "Balanced", "candidate_index": 1}]}
    strict.verify_fit(fitted, weights, global_weights, np.array([1, 2]))
    broken = deepcopy(fitted); broken["menu"][0]["weights"][0] = 0
    with unittest.TestCase().assertRaises(RuntimeError): strict.verify_fit(broken, weights, global_weights, np.array([1, 2]))


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for key, value in globals().items() if key.startswith("test_") and callable(value))


if __name__ == "__main__": unittest.main()
