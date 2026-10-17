"""Known post-fee observation sensitivity of frozen annual static policies."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research"), str(ROOT / "scripts")]
import run_risk_menu_research as primary
from safeinvest.data import load_paths
from safeinvest_research.metrics import NUMERIC_TOL, discrete_cvar, path_metrics
from safeinvest_research.static import simulate_annual_targets

PROTOCOL = ROOT / "docs/fee_checkpoint_diagnostic_protocol.json"
OUT = ROOT / "results/model_review/fee_checkpoint_v1"


def source_manifest():
    files = [Path(__file__).resolve(), PROTOCOL, primary.PROTOCOL, Path(primary.__file__).resolve(),
             ROOT / "src/safeinvest/core.py", ROOT / "src/safeinvest/data.py",
             ROOT / "research/safeinvest_research/static.py", ROOT / "research/safeinvest_research/metrics.py"]
    return {str(p.relative_to(ROOT)): primary.sha(p) for p in files}


def input_manifest():
    files = [primary.OUT / name for name in ["full_development_menus.json", "fit_manifest.json", "validation_fold_menus.json",
                                            "validation_manifest.json", "validation_oof_paths.csv"]]
    cache = primary.OUT / "cache" / f"{primary.cache_key(.05, .0025)}.npz"
    files += [cache, cache.with_suffix(".json")]
    return {str(p.relative_to(ROOT)): primary.sha(p) for p in files}


def fee_observation_mdd(nominal_wealth, fees, inflation):
    """Add observations, without altering or charging an executed path again.

    Wealth=(n,m,H+1), fees=(n,m,H), inflation=(n,H). Fee t is observed at
    transaction-time price P_(t-1), not at the end-of-year price P_t.
    """
    wealth, fee, infl = map(lambda value: np.asarray(value, float), (nominal_wealth, fees, inflation))
    if (wealth.ndim != 3 or wealth.shape[-1] < 2 or fee.shape != wealth.shape[:2] + (wealth.shape[-1] - 1,)
            or infl.shape != (wealth.shape[0], wealth.shape[-1] - 1) or not np.all(np.isfinite(wealth))
            or not np.all(np.isfinite(fee)) or not np.all(np.isfinite(infl)) or np.any(wealth <= 0)
            or np.any(fee < 0) or np.any(fee >= wealth[..., :-1]) or np.any(infl <= -1)):
        raise ValueError("Require positive executed wealth, matching nonnegative affordable fees and inflation")
    prices = np.column_stack((np.ones(len(infl)), np.cumprod(1 + infl, axis=1)))
    real = wealth / prices[:, None, :]
    annual = path_metrics(real)["mdd"]
    annual_peak = np.maximum.accumulate(real, axis=-1)
    real_fee = fee / prices[:, None, :-1]
    post_fee = real[..., :-1] - real_fee
    if np.any(post_fee > annual_peak[..., :-1] + 1e-12):
        raise RuntimeError("A nonnegative fee cannot create a new annual peak")
    fee_drawdown = np.maximum((annual_peak[..., :-1] - post_fee) / annual_peak[..., :-1], 0.)
    updated = np.maximum(annual, fee_drawdown.max(axis=-1))
    increment = updated - annual
    fee_to_prior_peak = real_fee / annual_peak[..., :-1]
    if np.any(increment > fee_to_prior_peak.max(axis=-1) + 2e-14):
        raise RuntimeError("Observation MDD increment exceeds the available fee contribution")
    return {"annual_mdd": annual, "feeobs_mdd": updated, "mdd_increment": increment,
            "post_fee_real": post_fee, "fee_point_drawdown": fee_drawdown,
            "real_wealth": real, "annual_peak": annual_peak,
            "fee_to_prior_peak": fee_to_prior_peak}


def summarize_pair(annual, updated, loss):
    annual, updated = np.asarray(annual, float), np.asarray(updated, float)
    if annual.ndim != 1 or updated.shape != annual.shape or not len(annual):
        raise ValueError("Require equally sized nonempty paired MDD vectors")
    old = annual > loss + NUMERIC_TOL; new = updated > loss + NUMERIC_TOL
    old_excess = np.where(old, annual - loss, 0.); new_excess = np.where(new, updated - loss, 0.)
    if np.any(old & ~new) or np.any(updated < annual) or np.any(new_excess < old_excess - 1e-14):
        raise RuntimeError("Adding fee observations cannot reduce path risk")
    return {"n": len(annual), "annual_breach_count": int(old.sum()), "feeobs_breach_count": int(new.sum()),
            "newly_flagged_paths": int((new & ~old).sum()),
            "annual_breach_probability": float(old.mean()), "feeobs_breach_probability": float(new.mean()),
            "mean_mdd_increment": float((updated - annual).mean()), "max_mdd_increment": float((updated - annual).max()),
            "paths_with_positive_increment": int(np.count_nonzero(updated > annual + NUMERIC_TOL)),
            "annual_median_mdd": float(np.median(annual)), "feeobs_median_mdd": float(np.median(updated)),
            "annual_p90_mdd": float(np.quantile(annual, .90, method="inverted_cdf")),
            "feeobs_p90_mdd": float(np.quantile(updated, .90, method="inverted_cdf")),
            "annual_expected_excess": float(old_excess.mean()), "feeobs_expected_excess": float(new_excess.mean()),
            "annual_cvar90": float(discrete_cvar(annual)), "feeobs_cvar90": float(discrete_cvar(updated))}


def main():
    started = time.perf_counter()
    sources = source_manifest()
    protocol = json.loads(PROTOCOL.read_text())
    research = json.loads(primary.PROTOCOL.read_text())
    arrays, cached = primary.get_cache(research, step=.05, cost=.0025)
    inputs = input_manifest()
    returns, inflation, ids = load_paths(ROOT / "data/SafeInvest_Synthetic_Data.csv", range(1, 121), allow_final=False)
    if not np.array_equal(ids, np.arange(1, 121)) or not np.array_equal(ids, arrays["scenario_ids"]) or len(arrays["weights"]) != 232:
        raise RuntimeError("Only the frozen 232-target development family is permitted")
    digest = hashlib.sha256()
    for value in (ids, returns, inflation): digest.update(np.ascontiguousarray(value).tobytes())
    cache_meta = primary.OUT / "cache" / f"{primary.cache_key(.05,.0025)}.json"
    if digest.hexdigest() != json.loads(cache_meta.read_text())["development_arrays_sha256"]:
        raise RuntimeError("Executed development inputs differ from guarded cache")
    simulation = simulate_annual_targets(returns, inflation, arrays["weights"], .0025)
    observation, cache_checks, grid_rows, grid_paths = {}, [], [], []
    for horizon in protocol["horizons"]:
        wealth = simulation["nominal_wealth"][..., :horizon + 1]
        fees = simulation["fees"][..., :horizon]
        observed = fee_observation_mdd(wealth, fees, inflation[:, :horizon])
        observation[horizon] = observed
        metrics = path_metrics(observed["real_wealth"])
        errors = {}
        for key, value in metrics.items():
            np.testing.assert_allclose(value, cached[horizon][key], atol=0, rtol=0)
            errors[key] = float(np.abs(value - cached[horizon][key]).max())
        for key in ["fees", "moved", "trade_fraction"]:
            actual = (simulation["trades"][..., :horizon].mean(axis=-1) if key == "trade_fraction"
                      else simulation[key][..., :horizon].sum(axis=-1))
            np.testing.assert_allclose(actual, arrays[f"H{horizon}__{key}"], atol=0, rtol=0)
            errors[key] = float(np.abs(actual - arrays[f"H{horizon}__{key}"]).max())
        pretrade = simulation["nominal_wealth"][..., :horizon]
        if np.any(fees / pretrade > .0025 / 1.0025 + 1e-14):
            raise RuntimeError("Fee fraction exceeds the model's full-switch upper bound")
        cache_checks.append({"horizon": horizon, "cache_max_abs_errors": errors,
                             "max_feeobs_minus_annual_mdd": float(observed["mdd_increment"].max()),
                             "fee_point_never_raises_peak": True})
    for client in research["clients"]:
        obs = observation[client["horizon"]]
        for candidate, weights in enumerate(arrays["weights"]):
            old, new = obs["annual_mdd"][:, candidate], obs["feeobs_mdd"][:, candidate]
            grid_rows.append({"client": client["name"], "horizon": client["horizon"], "loss_tolerance": client["loss_tolerance"],
                              "candidate_index": candidate, "w_equity": weights[0], "w_bond": weights[1], "w_bill": weights[2],
                              **summarize_pair(old, new, client["loss_tolerance"])})
            for row, scenario in enumerate(ids):
                grid_paths.append({"scenario_id": int(scenario), "client": client["name"], "candidate_index": candidate,
                                   "horizon": client["horizon"], "loss_tolerance": client["loss_tolerance"],
                                   "annual_mdd": float(old[row]), "feeobs_mdd": float(new[row]),
                                   "mdd_increment": float(new[row] - old[row]),
                                   "annual_breach": bool(old[row] > client["loss_tolerance"] + NUMERIC_TOL),
                                   "feeobs_breach": bool(new[row] > client["loss_tolerance"] + NUMERIC_TOL)})
    grid = pd.DataFrame(grid_rows)
    menus = primary.compatible_full_fit()
    full_pairs = []
    for case, fit in menus.items():
        for detail in fit["client_product_details"]:
            row = grid.loc[(grid.client == detail["client"]) & (grid.candidate_index == detail["candidate_index"])].iloc[0].to_dict()
            full_pairs.append({"case": case, "requested_alpha": fit["alpha"], "portfolio": detail["portfolio"],
                               "chosen": detail["chosen"], "original_assignment": fit["assignments"][detail["client"]],
                               "original_sample_feasible": detail["sample_feasible"], **row,
                               "annual_empirical_risk_pass": row["annual_breach_probability"] <= fit["alpha"] + NUMERIC_TOL,
                               "feeobs_empirical_risk_pass": row["feeobs_breach_probability"] <= fit["alpha"] + NUMERIC_TOL})
    original = pd.read_csv(primary.OUT / "validation_oof_paths.csv")
    original = original.loc[original.case == "alpha_10__risk_only"]
    if len(original) != 600:
        raise RuntimeError("Require all 600 original primary OOF client-path records")
    oof_rows = []
    for source in original.to_dict("records"):
        client = next(p for p in research["clients"] if p["name"] == source["client"])
        row = {key: source[key] for key in ["fold", "scenario_id", "client", "assignment", "assigned", "candidate_index"]}
        row.update({"horizon": client["horizon"], "loss_tolerance": client["loss_tolerance"],
                    "annual_mdd": None, "feeobs_mdd": None, "mdd_increment": None,
                    "annual_breach": None, "feeobs_breach": None, "annual_excess": None, "feeobs_excess": None,
                    "original_log_growth_unchanged": None})
        if source["assigned"]:
            candidate, scene = int(source["candidate_index"]), int(source["scenario_id"]) - 1
            obs = observation[client["horizon"]]
            old, new = float(obs["annual_mdd"][scene, candidate]), float(obs["feeobs_mdd"][scene, candidate])
            if not np.isclose(old, source["mdd"], atol=1e-14, rtol=0):
                raise RuntimeError("Annual OOF risk changed before adding observations")
            if not np.isclose(cached[client["horizon"]]["log_growth"][scene, candidate], source["log_growth"], atol=1e-14, rtol=0):
                raise RuntimeError("Original OOF growth differs from unchanged executed wealth")
            breach_old, breach_new = old > client["loss_tolerance"] + NUMERIC_TOL, new > client["loss_tolerance"] + NUMERIC_TOL
            row.update({"annual_mdd": old, "feeobs_mdd": new, "mdd_increment": new - old,
                        "annual_breach": breach_old, "feeobs_breach": breach_new,
                        "annual_excess": old - client["loss_tolerance"] if breach_old else 0.,
                        "feeobs_excess": new - client["loss_tolerance"] if breach_new else 0.,
                        "original_log_growth_unchanged": source["log_growth"]})
        else:
            row["candidate_index"] = None
        oof_rows.append(row)
    oof = pd.DataFrame(oof_rows)
    oof_summary = []
    for client, group in oof.groupby("client", sort=False):
        assigned = group.loc[group.assigned]
        row = {"client": client, "total_test_paths": len(group), "assigned_test_paths": len(assigned),
               "review_test_paths": int((~group.assigned).sum()), "diagnostic": None}
        if len(assigned):
            loss = float(assigned.loss_tolerance.iloc[0])
            row["diagnostic"] = summarize_pair(assigned.annual_mdd, assigned.feeobs_mdd, loss)
        oof_summary.append(row)
    if source_manifest() != sources or input_manifest() != inputs:
        raise RuntimeError("Source or frozen input changed; no result was written")
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {"grid_client_summary.csv": grid, "grid_client_paths.csv": pd.DataFrame(grid_paths),
              "full_menu_all_pairs.csv": pd.DataFrame(full_pairs), "primary_oof_paths.csv": oof}
    for file, frame in frames.items(): frame.to_csv(OUT / file, index=False)
    primary.write_json(OUT / "primary_oof_summary.json", oof_summary)
    primary.write_json(OUT / "accounting_checks.json", {"cache": cache_checks, "annual_growth_and_fees_unchanged": True,
                                                        "fee_only_adds_observation": True, "referral_rows_retained_null": int((~oof.assigned).sum())})
    files = [*frames, "primary_oof_summary.json", "accounting_checks.json"]
    end_sources, end_inputs = source_manifest(), input_manifest()
    if end_sources != sources or end_inputs != inputs:
        raise RuntimeError("Sources or frozen inputs changed while writing; no completion manifest")
    primary.write_json(OUT / "manifest.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter() - started,
        "sources": sources, "sources_at_end": end_sources, "frozen_input_hashes": inputs, "frozen_input_hashes_at_end": end_inputs,
        "development_arrays_sha256": digest.hexdigest(), "scenario_ids": ids.tolist(), "final_scenarios_loaded": False,
        "outputs_sha256": {file: primary.sha(OUT / file) for file in files}, "protocol": protocol,
        "scope": "Observation-definition sensitivity only; original annual MDD, strategies, assignments, growth and fees remain unchanged. Fee observations are not measured continuous/intrayear MDD.",
    })
    print(f"Wrote fee-checkpoint definition sensitivity to {OUT}", flush=True)


if __name__ == "__main__":
    main()
