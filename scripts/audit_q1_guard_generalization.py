"""Independent audit of the Q1 guard generalization run.

Reimplements every per-path metric and summary statistic directly from the
saved physical arrays, WITHOUT importing the producer's metrics or policy
modules. Also checks the self-financing accounting identity on every decision,
ID/horizon/L conventions, freeze integrity and output hashes. Any mismatch
fails the audit; the producer's own code is never trusted for verification.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/q1_guard_generalization_v1'
PROTOCOL = ROOT / 'docs/q1_guard_generalization_protocol.json'
FREEZE = ROOT / 'docs/q1_guard_generalization_source_freeze.json'
TOL = 1e-12
ATOL = 1e-9
DEV_IDS = list(range(1, 121))

CHECKS = 0
FAILURES = []


def check(condition, label):
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(label)
        raise RuntimeError('Audit check failed: ' + label)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with gzip.open(path, 'rt', newline='') as handle:
        return list(csv.DictReader(handle))


def independent_metrics(real_wealth):
    """Recompute growth/CAGR/MDD/volatility from real wealth, no repo imports."""
    w = np.asarray(real_wealth, float)
    n, hp1 = w.shape
    h = hp1 - 1
    check(np.all(np.isfinite(w)) and np.all(w > 0), 'real wealth finite positive')
    check(np.all(np.abs(w[:, 0] - 100.0) <= ATOL), 'initial real wealth is 100')
    log_growth = np.log(w[:, -1] / 100.0) / h
    cagr = np.expm1(log_growth)
    peaks = np.maximum.accumulate(w, axis=1)
    mdd = ((peaks - w) / peaks).max(axis=1)
    rets = w[:, 1:] / w[:, :-1] - 1.0
    volatility = np.std(rets, axis=1, ddof=1) if h > 1 else np.zeros(n)
    return dict(log_growth=log_growth, cagr=cagr, mdd=mdd, volatility=volatility,
                terminal_real=w[:, -1])


def independent_quantile(values, p):
    v = sorted(values)
    return v[max(0, math.ceil(p * len(v)) - 1)]


def independent_median(values):
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def independent_cvar90(losses):
    n = len(losses)
    mass = 0.1 * n
    return math.fsum(x * min(max(mass - i, 0.), 1.)
                     for i, x in enumerate(sorted(losses, reverse=True))) / mass


def close(a, b):
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= ATOL * max(1.0, abs(float(a)), abs(float(b)))


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    check(manifest['status'] == 'complete', 'producer manifest complete')
    check(manifest['final_scenarios_loaded'] is False, 'no final scenarios loaded')
    check(manifest.get('new_final_reads') == 0, 'zero new final reads declared')

    protocol = json.loads(PROTOCOL.read_text())
    freeze = json.loads(FREEZE.read_text())
    check(freeze['protocol_sha256'] == sha(PROTOCOL), 'protocol hash matches freeze')
    check(manifest['protocol_sha256'] == sha(PROTOCOL), 'manifest binds protocol')
    for key in ('source_sha256', 'input_sha256'):
        for name, digest in freeze[key].items():
            check(sha(ROOT / name) == digest, 'frozen file unchanged: ' + name)
            check(manifest['sources_and_inputs_at_start'].get(name) == digest,
                  'manifest start hash binds: ' + name)
    check(manifest['sources_and_inputs_at_end'] == manifest['sources_and_inputs_at_start'],
          'no source/input changed during run')
    for name, digest in manifest['outputs_sha256'].items():
        check(sha(OUT / name) == digest, 'output hash unchanged: ' + name)

    traces = np.load(OUT / 'traces.npz')
    rows = read_csv(OUT / 'path_metrics.csv.gz')
    summaries = read_csv(OUT / 'summary.csv.gz')
    check(len(rows) == 960, '960 path rows')
    check(len(summaries) == 8, '8 summary rows')

    objects = {o['object_id']: o for o in protocol['objects']}
    check(set(objects) == {'B_guard', 'B_same_target_annual', 'B_same_target_pending',
                           'B_exact_thirds_annual', 'E_guard', 'E_same_target_annual',
                           'E_same_target_pending', 'E_exact_thirds_annual'}, 'object inventory')

    for oid, obj in objects.items():
        h, L = obj['horizon'], obj['L']
        orows = [r for r in rows if r['object_id'] == oid]
        check(len(orows) == 120, oid + ': 120 rows')
        check(sorted(int(r['scenario_id']) for r in orows) == DEV_IDS, oid + ': exact IDs 1-120 once')
        funded = all(r['assigned'] == 'True' for r in orows)
        if not funded:
            # Honest Review arm: every financial field NULL, no physical arrays archived.
            check(all(r['assigned'] == 'False' for r in orows), oid + ': uniform review status')
            for r in orows:
                for k in ('log_growth', 'cagr', 'mdd', 'volatility', 'terminal_real',
                          'nominal_fees', 'trade_fraction', 'breach', 'first_crossing',
                          'automatic_fraction', 'protection_trade_fraction', 'support_review',
                          'mitigation_trade_fraction', 'mitigation_nominal_fees'):
                    check(r[k] in ('', None), oid + f': NULL {k} for unfunded row')
                check(r['execution_status'] == 'preinvestment_review', oid + ': review status label')
            check(oid + '__real_wealth' not in traces, oid + ': no invented physical arm')
            srow = next(s for s in summaries if s['object_id'] == oid)
            check(int(srow['financial_n']) == 0 and int(srow['original_n']) == 120, oid + ': zero funded n')
            check(srow['breach_count'] in ('', None) and srow['median_cagr'] in ('', None),
                  oid + ': NULL summary financials')
            continue
        rw = traces[oid + '__real_wealth']
        check(rw.shape == (120, h + 1), oid + ': own-horizon shape')
        fees = traces[oid + '__fees']
        moved = traces[oid + '__moved']
        trades = traces[oid + '__trades']
        post = traces[oid + '__posttrade_holdings']
        pretrade = traces[oid + '__pretrade_wealth']
        m = independent_metrics(rw)

        # Self-financing identity on every decision: posttrade sum + fee == pretrade wealth.
        check(np.all(np.abs(post.sum(axis=2) + fees - pretrade) <= 1e-6 * np.maximum(pretrade, 1.0)),
              oid + ': self-financing identity')
        # Fee rate convention: fee == 0.0025 * moved.
        check(np.all(np.abs(fees - 0.0025 * moved) <= 1e-9 * np.maximum(fees, 1.0)),
              oid + ': fee equals 0.25% of moved')
        # Trades flag equals moved > 1e-12.
        check(np.all(trades == (moved > TOL)), oid + ': trade flag convention')
        # Breach convention from running MDD.
        running = traces[oid + '__running_mdd']
        ever = traces[oid + '__ever_breached']
        check(np.all(ever[:, -1] == (running[:, -1] - L > TOL)), oid + ': breach convention')
        peaks = np.maximum.accumulate(rw, axis=1)
        drawdowns = (peaks - rw) / peaks
        check(np.all(np.abs(running - np.maximum.accumulate(drawdowns, axis=1)) <= 1e-9),
              oid + ': running MDD equals running max drawdown')

        for i, r in enumerate(orows):
            sid = int(r['scenario_id'])
            check(sid == DEV_IDS[i], oid + ': row order matches IDs')
            check(close(float(r['cagr']), m['cagr'][i]), oid + f': cagr row {sid}')
            check(close(float(r['log_growth']), m['log_growth'][i]), oid + f': log_growth row {sid}')
            check(close(float(r['mdd']), m['mdd'][i]), oid + f': mdd row {sid}')
            check(close(float(r['volatility']), m['volatility'][i]), oid + f': volatility row {sid}')
            check(close(float(r['terminal_real']), m['terminal_real'][i]), oid + f': terminal row {sid}')
            check(close(float(r['nominal_fees']), fees[i].sum()), oid + f': fees row {sid}')
            check(close(float(r['trade_fraction']), trades[i].mean()), oid + f': trade fraction row {sid}')
            check((r['breach'] == 'True') == bool(ever[i, -1]), oid + f': breach row {sid}')
            fc = int(r['first_crossing']) if r['first_crossing'] else None
            # Producer convention: first_crossing is the 1..H endpoint index of first breach.
            expected_fc = int(np.argmax(ever[i])) if ever[i].any() else None
            check(fc == expected_fc, oid + f': first crossing row {sid}')

        srow = next(s for s in summaries if s['object_id'] == oid)
        cagrs = [float(r['cagr']) for r in orows]
        losses = [float(r['mdd']) for r in orows]
        vols = [float(r['volatility']) for r in orows]
        fee_list = [float(r['nominal_fees']) for r in orows]
        tf = [float(r['trade_fraction']) for r in orows]
        af = [float(r['automatic_fraction']) for r in orows]
        check(int(srow['breach_count']) == sum(r['breach'] == 'True' for r in orows), oid + ': breach count')
        check(close(float(srow['median_cagr']), independent_median(cagrs)), oid + ': median cagr')
        check(close(float(srow['p10_cagr']), independent_quantile(cagrs, 0.1)), oid + ': p10 cagr')
        check(close(float(srow['median_mdd']), independent_median(losses)), oid + ': median mdd')
        check(close(float(srow['p90_mdd']), independent_quantile(losses, 0.9)), oid + ': p90 mdd')
        check(close(float(srow['cvar90_mdd']), independent_cvar90(losses)), oid + ': cvar90')
        check(close(float(srow['mean_mdd']), math.fsum(losses) / 120), oid + ': mean mdd')
        check(close(float(srow['mean_volatility']), math.fsum(vols) / 120), oid + ': mean volatility')
        check(close(float(srow['mean_nominal_fees']), math.fsum(fee_list) / 120), oid + ': mean fees')
        check(close(float(srow['mean_trade_fraction']), math.fsum(tf) / 120), oid + ': mean trade fraction')
        check(close(float(srow['mean_automatic_fraction']), math.fsum(af) / 120), oid + ': mean automatic')

    comparisons = json.loads((OUT / 'comparisons.json').read_text())
    check(len(comparisons) == 4, 'four paired comparisons')
    for comp in comparisons:
        ids = comp['same_funded_ids']
        label = comp['model_object'] + ' vs ' + comp['reference_object']
        if not ids:
            # Unfunded guard: no paired financial comparison may be invented.
            parts = comp['event_id_partition']
            check(all(not v for v in parts.values()), label + ': empty partition for unfunded model')
            check(all(v['mean'] is None for v in comp['paired_differences'].values()),
                  label + ': NULL paired differences for unfunded model')
            check(comp['model']['financial_n'] == 0 and comp['reference']['financial_n'] == 120,
                  label + ': model unfunded, reference intact')
            continue
        check(ids == DEV_IDS, label + ': full pairing')
        parts = comp['event_id_partition']
        check(sorted(parts['neither'] + parts['model_only'] + parts['reference_only'] + parts['both']) == DEV_IDS,
              label + ': breach partition covers all IDs once')

    result = dict(status='passed' if not FAILURES else 'failed', checks=CHECKS,
                  failures=FAILURES, audited_output=str(OUT.relative_to(ROOT)),
                  scope='Independent recomputation from saved arrays; no producer metric/policy import.',
                  note='Implementation/numerical audit only; not statistical validation or market evidence.')
    (OUT / 'independent_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    sys.exit(main())
