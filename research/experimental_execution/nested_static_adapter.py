"""Training-only adapter for the frozen strict-positive static menu learner.

No function reads a dataset, cache, result file or protocol. The caller supplies
precomputed annual-policy metrics with original scenario ID i at row i-1 and
explicitly freezes all seeds/fold identities before any market fitting. Metrics
outside the nominated training IDs never participate, including in validation.

The original raw five-profile gamma objective and post-refit admission guard are
reused unchanged. Balanced-only is a separate downstream Q2 case classification;
it never changes the Q1 menu objective or rewrites an effective assignment.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import run_nested_static_calibration as frozen_nested
import run_positive_asset_specification as frozen_strict
from safeinvest_research.static import simplex_grid

GAMMAS = frozen_nested.GAMMAS
SERVICE_ALPHA = frozen_nested.SERVICE_ALPHA
STATIC_FIELDS = ("log_growth", "cagr", "mdd", "volatility", "terminal_real")
SUPPORTED_TRAINING_SIZES = (72, 90, 96, 120)


def _ids(values, name, *, empty=False):
    try:
        raw = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence") from exc
    if any(isinstance(i, (bool, np.bool_)) for i in raw):
        raise ValueError(f"{name} cannot contain booleans")
    values = np.asarray(raw)
    if (values.ndim != 1 or (not len(values) and not empty)
            or (len(values) and (not np.issubdtype(values.dtype, np.integer)
                                 or np.issubdtype(values.dtype, np.bool_)))):
        raise ValueError(f"{name} must be unique original integer scenario IDs")
    result = [int(i) for i in values]
    if len(set(result)) != len(result) or any(i < 1 or i > 120 for i in result):
        raise ValueError(f"{name} must be unique original IDs in 1..120; final IDs are forbidden")
    return sorted(result)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def make_context(training_ids, *, seed, name="static_training", context_id=None,
                 outer_test_ids=()):
    """Create four explicitly seeded, complementary static subfolds inside T.

    T must contain 72, 90, 96 or 120 unique original IDs, each in 1..120.
    Permute sorted(T) once with default_rng(seed), array_split into four chunks,
    and store each test chunk and its training complement in sorted ID order.
    For T=90 this deliberately gives test sizes23/23/22/22 and train67/67/68/68.
    No seed is derived or silently defaulted. New dynamic-inner tasks use the
    caller's protocol seed; legacy six tasks use their recorded seed and folds.
    """
    train = _ids(training_ids, "training_ids")
    if len(train) not in SUPPORTED_TRAINING_SIZES:
        raise ValueError("Require 72, 90, 96 or 120 explicit training scenarios")
    outer = _ids(outer_test_ids, "outer_test_ids", empty=True)
    if set(train) & set(outer):
        raise ValueError("Training and outer evaluation IDs must be disjoint")
    seed = _integer(seed, "seed")
    if not 0 <= seed < 2**64:
        raise ValueError("seed must lie in [0,2**64)")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a nonempty string")
    if context_id is not None:
        context_id = _integer(context_id, "context_id")
        if context_id < 0:
            raise ValueError("context_id must be nonnegative or None")
    folds = []
    for number, chunk in enumerate(np.array_split(np.random.default_rng(seed).permutation(train), 4), 1):
        test = sorted(map(int, chunk))
        folds.append({"inner_fold": number, "train_ids": sorted(set(train) - set(test)),
                      "test_ids": test})
    return {"context_id": context_id, "name": name, "training_ids": train,
            "outer_test_ids": outer, "inner_seed": seed, "inner_folds": folds}


def validate_context(context):
    """Require exact pre-recorded fold identities, including unequal T90 folds."""
    if not isinstance(context, Mapping):
        raise ValueError("context must be a mapping")
    try:
        expected = make_context(context["training_ids"], seed=context["inner_seed"],
                                name=context["name"], context_id=context["context_id"],
                                outer_test_ids=context["outer_test_ids"])
        actual_folds = context["inner_folds"]
    except KeyError as exc:
        raise ValueError(f"Missing context field: {exc.args[0]}") from exc
    if not isinstance(actual_folds, list) or len(actual_folds) != 4:
        raise ValueError("Require exactly four static subfolds")
    for actual, planned in zip(actual_folds, expected["inner_folds"], strict=True):
        try:
            fold = _integer(actual["inner_fold"], "inner_fold")
            train = _ids(actual["train_ids"], "inner train_ids")
            test = _ids(actual["test_ids"], "inner test_ids")
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed static fold") from exc
        if fold != planned["inner_fold"] or train != planned["train_ids"] or test != planned["test_ids"]:
            raise ValueError("Static folds differ from the explicitly seeded training-only partition")
    return expected


def prepare_strict_family(weights232, metrics_by_horizon):
    """Subset the unchanged232 .05-simplex+thirds cache to the frozen172 family.

    Return (original232_indices, weights172, metrics172). This performs no fit.
    Rows retain the original ID i -> row i-1 convention. Values are copied by
    the frozen subset helper; the caller retains responsibility for guarded
    cache/data/source hashes. No new weight or risk threshold is introduced.
    """
    weights = np.asarray(weights232, float)
    if not np.array_equal(weights, simplex_grid(.05)):
        raise ValueError("Require the exact frozen232 candidate ordering and weights")
    allowed, subset, metrics = frozen_strict.subset_family(weights, metrics_by_horizon)
    if len(allowed) != 172:
        raise RuntimeError("Frozen strict-positive family must contain172 candidates")
    for fields in metrics.values():
        if any(values.shape[0] != 120 for values in fields.values()):
            raise ValueError("Static cache rows must be original scenarios1..120")
    return allowed, subset, metrics


def _validate_inputs(metrics, weights, profiles, train):
    weights = np.asarray(weights, float)
    if (weights.ndim != 2 or weights.shape[1] != 3 or len(weights) < 3
            or not np.all(np.isfinite(weights)) or np.any(weights < .05 - 1e-12)
            or not np.all(np.abs(weights.sum(axis=1) - 1) <= 1e-12)):
        raise ValueError("Require at least three fully invested strict-positive targets")
    if not isinstance(profiles, (list, tuple)) or not profiles:
        raise ValueError("Require explicit questionnaire profiles")
    names, horizons = [], {15}  # frozen menu-completion reference horizon
    for p in profiles:
        try:
            name, h, loss, liquidity = p["name"], p["horizon"], p["loss_tolerance"], p["liquidity"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed questionnaire profile") from exc
        if (not isinstance(name, str) or not name or name in names
                or isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer))
                or not 1 <= h <= 30 or isinstance(loss, (bool, np.bool_))
                or not np.isfinite(loss) or not 0 <= loss <= 1
                or liquidity not in frozen_nested.FLOORS):
            raise ValueError("Invalid questionnaire profile")
        names.append(name); horizons.add(int(h))
    indices = np.asarray(train, dtype=int) - 1
    for h in horizons:
        if h not in metrics:
            raise ValueError(f"Missing static metrics for horizon{h}")
        for field in STATIC_FIELDS:
            if field not in metrics[h]:
                raise ValueError(f"Missing static metric: {field}")
            values = np.asarray(metrics[h][field])
            if values.shape != (120, len(weights)):
                raise ValueError("Metrics require original120 rows and the supplied candidate columns")
            if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values[indices])):
                raise ValueError("Metrics must be finite on nominated training IDs only")
    return weights


def classify_balanced_case(refit, client="C"):
    """Classify Q2 after Q1 effective admission without changing Q1 evidence.

    product is present only for a Balanced case; other cases have no Q2
    financial outcome. q1_product separately preserves any Cons/Growth product.
    This is case membership, not dynamic execution admission.
    """
    try:
        assignment = refit["effective_assignments"][client]
    except (KeyError, TypeError) as exc:
        raise ValueError("Client absent from effective Q1 assignments") from exc
    fitted = refit.get("raw_fit")
    menu = fitted["menu"] if fitted else []
    product = next((p for p in menu if p["name"] == assignment), None)
    if assignment != "Review" and product is None:
        raise ValueError("Effective Q1 assignment lacks a recorded product")
    if assignment == "Review":
        status = "risk_review"
    elif assignment == "Balanced":
        status = "balanced_case"
    elif assignment in ("Conservative", "Growth"):
        status = "not_balanced_case"
    else:
        raise ValueError("Unknown effective Q1 portfolio label")
    return {"client": client, "status": status, "effective_assignment": assignment,
            "in_balanced_case": status == "balanced_case",
            "product": deepcopy(product) if status == "balanced_case" else None,
            "q1_product": deepcopy(product),
            "execution_admission_not_yet_evaluated": status == "balanced_case"}


def calibrate_static(metrics_by_horizon, weights, profiles, context, *,
                     candidate_original_indices=None):
    """Fully reselect static gamma/menu/admission using this context's T only.

    Return JSON-compatible keys refit, inner_summaries, inner_fold_fits,
    inner_oof_paths, q2_case, context and metadata. Only the four static test
    chunks inside T score gamma. No rows outside T enter gamma/menu/admission.
    The five-profile raw objective, coverage/growth/gamma ordering, and final
    zero-inner-auto admission guard are the unchanged frozen implementation.

    weights may be a smaller strict-positive synthetic family for unit tests;
    production must use prepare_strict_family's172 candidates and mapping.
    candidate_index always remains local. If mapping is supplied, additional
    original232 fields are attached through the frozen annotation helper.
    """
    canonical = validate_context(context)
    weights = _validate_inputs(metrics_by_horizon, weights, profiles, canonical["training_ids"])
    if candidate_original_indices is not None:
        mapping = np.asarray(candidate_original_indices)
        if (mapping.shape != (len(weights),) or not np.issubdtype(mapping.dtype, np.integer)
                or np.issubdtype(mapping.dtype, np.bool_) or np.any(mapping < 0)
                or np.any(mapping >= 232) or len(set(map(int, mapping))) != len(mapping)):
            raise ValueError("Require unique local-to-original232 integer mapping")
    else:
        mapping = None
    summaries, records, folds_by_gamma = [], [], {}
    for gamma in GAMMAS:
        rows, fits = [], []
        for fold in canonical["inner_folds"]:
            fitted = frozen_nested.fit_gamma(metrics_by_horizon, weights, profiles,
                                             np.asarray(fold["train_ids"]) - 1, gamma)
            fitted["inner_fold"] = fold["inner_fold"]
            fitted["train_ids"] = fold["train_ids"]
            fitted["test_ids"] = fold["test_ids"]
            fits.append(fitted)
            rows.extend(frozen_nested.path_records(metrics_by_horizon, profiles, fitted, fold["test_ids"],
                        context=canonical["name"], gamma=gamma, inner_fold=fold["inner_fold"]))
        summaries.append(frozen_nested.inner_summary(rows, gamma))
        records.extend(rows); folds_by_gamma[str(gamma)] = fits
    selection = frozen_nested.select_gamma(summaries)
    gamma = selection["selected_gamma"]
    chosen = next((s for s in summaries if s["training_screen_gamma"] == gamma), None)
    raw_fit = None
    if gamma is not None:
        raw_fit = frozen_nested.fit_gamma(metrics_by_horizon, weights, profiles,
                                          np.asarray(canonical["training_ids"]) - 1, gamma)
        # Preserve the historical conservative1160 fixed-static-family
        # diagnostic for old-context reproduction. It never selects/admit Q2.
        counts = np.asarray([r["breach_count"] for r in raw_fit["client_product_details"]], dtype=int)
        upper = frozen_nested.simultaneous_upper(counts, len(canonical["training_ids"]), 1160, .05)
        for detail, bound in zip(raw_fit["client_product_details"], upper, strict=True):
            detail["postfit_static_family_risk_upper"] = float(bound)
            detail["postfit_bound_within_service_alpha"] = bool(bound <= SERVICE_ALPHA)
            detail["postfit_bound_used_in_selection"] = False
    refit = {"context": canonical["name"], "context_id": canonical["context_id"],
             "training_ids": canonical["training_ids"], "outer_test_ids": canonical["outer_test_ids"],
             "selection": selection, "selected_gamma_inner_evidence": chosen, "raw_fit": raw_fit,
             **frozen_nested.apply_admission(raw_fit, profiles, chosen)}
    output = {"refit": refit, "inner_summaries": summaries, "inner_fold_fits": folds_by_gamma,
              "inner_oof_paths": records}
    if mapping is not None:
        output = frozen_strict.annotate_indices(output, mapping)
    output["q2_case"] = classify_balanced_case(output["refit"]) if any(p["name"] == "C" for p in profiles) else None
    output["context"] = deepcopy(canonical)
    output["metadata"] = {"training_only": True, "market_data_loaded": False,
        "scenario_row_convention": "original scenario i at row i-1, rows1..120 only",
        "candidate_count": len(weights), "candidate_index_space": "local; optional additional original232 fields",
        "service_alpha": SERVICE_ALPHA, "gamma_candidates": list(GAMMAS),
        "liquidity_case": "risk_only", "static_gamma_objective": "unchanged raw all-profile inner OOF",
        "balanced_only_applied_after_Q1": True,
        "inner_fold_aggregation": "pool each original path once, not an unweighted mean of fold statistics",
        "postfit_bound_used_in_selection": False,
        "risk_certificate": False}
    return output


def reproduce_legacy_context(metrics232, weights232, profiles, context):
    """Same-fold adapter entry point for a frozen full120/outer96 context.

    The caller supplies its recorded legacy context; this function neither
    reads old results nor fits unless explicitly called. Only the legacy fold
    validator is additionally required. New90/72 tasks use calibrate_static.
    """
    frozen_nested.validate_context(context)
    allowed, weights, metrics = prepare_strict_family(weights232, metrics232)
    return calibrate_static(metrics, weights, profiles, context,
                            candidate_original_indices=allowed)
