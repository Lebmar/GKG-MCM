# Team 87 — compact technical source

Participant: Chengxuan YUAN. Team: Lebmar.

Use `scripts/submission_adapter.py` for the compact reproduction. It calls the
unchanged locked V2b kernel. V3 and earlier programs are retained as exploratory
and historical source, and are not substituted for the locked main model.
No paper or historical large result archive is included in this source package.

Install the versions in `requirements-research-pinned.txt`, then run:

```sh
python scripts/submission_adapter.py
```

The default uses generated synthetic arrays only, runs a small full-horizon
smoke example, and writes a fresh `submission_reproduction_synthetic_v1`
directory. It does not open the supplied market CSV.

After a compact package has been prepared and its manifest verified, explicitly
reproduce the fixed development objects with:

```sh
python scripts/submission_adapter.py --development
```

This reads original scenario IDs 1–120 only. It compares the full fixed main
model, same-target annual-then-pending B, same-target annual restoration, and
exact-thirds annual restoration against small canonical per-field
dtype/shape/value digests and financial summaries. No fitting, parameter
selection, alternate candidate or final-data mode exists. Each command refuses
to overwrite a previous complete, failed or incomplete output; use a new
`--output` path for a separate explicitly intended run.

The compact adapter is a new fixed-model reproduction entrypoint. The old
historical runners retain their original large transitive evidence checks;
their complete historical NPZ/cache archives are not contained in this ZIP.
Snapshot reproduction is not a rerun of all nested learning or new independent
market validation. The original source and declared limitations are retained
for inspection.

The provided synthetic CSV is unmodified. See `docs/DATA_NOTICE.md`,
`docs/references.md` and `submission/LICENSE-DATA-CC-BY-NC-SA-4.0.txt` for the
competition/JST attribution and CC BY-NC-SA 4.0 data conditions. The data are
synthetic scenarios, not historical observations or forecasts. No new blanket
open-source licence is granted to the participant's original code by this
package. See `submission/AI_USE.md` for the actual scope of AI assistance and
the participant's responsibility to review the work.
