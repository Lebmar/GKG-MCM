# Additional development diagnostics, 19 October 2026

These additions leave the selected V2b model, its parameters, the previously
exposed final evaluation, and the existing V2 source ZIP unchanged. They are
technical evidence for discussion, not the competition paper.

Start with `notebooks/technical_last_mile_2026_10_19.html`. Its four figures retain
PNG, SVG and source CSV files. The page uses only the original 120 development
scenarios, with each outer fold's own target and selected controller. Full-sample
resubstitution remains separate in the underlying support diagnostic.

The new checks cover:

- Review timing, continuing exposure, exact fee/return accounting, five explicitly
  hypothetical service-fee scenarios, and eight prespecified lower-exposure
  references. References have no new admission certification or winner selection.
- Calendar versus pooled joint support and base/stressed nominal-path × CPI
  cross-valuation. All peaks and drawdowns retain the original investment history.
- Eighteen original-problem requirements, exact parameter meanings and remaining
  operational/validation limits.

Independent numerical audits are saved alongside the two diagnostic producers.
They verify the calculations; they do not provide new market observations or
restore blind validation. The failed first support-diagnostic attempt remains in
`results/support_failure_diagnostic_v1/`; the corrected metadata adapter writes to
`results/support_failure_diagnostic_v2/` without changing statistical definitions.

## Reproduction

`scripts/reproduce_last_mile_diagnostics.py --execute` runs both diagnostic
producers in separate fresh workspaces using the existing `.venv/bin/python -I`.
It copies only each producer's frozen dependencies, never seeds the target
diagnostic outputs, and compares all declared outputs against the originals.
It verifies actual module origins, every NPZ field/dtype/shape/value and the NPZ
container hash. Only gzip-header timestamps and the declared root-manifest UTC
fields may differ; gzip contents and all other bytes must match.

The checked run is retained in
`results/technical_last_mile_reproduction_v1/manifest.json`. The script refuses
to overwrite it. To run again, use a separate copy of the full repository with a
working compatible `.venv`, omit **only that reproduction-attempt directory** from
the copy, and run from the copied repository root:

```sh
.venv/bin/python scripts/reproduce_last_mile_diagnostics.py --execute
```

Keep both original diagnostic output directories and their independent audits in
that copy: they are comparison evidence, not input seeds for the isolated child
processes. Preserve the original repository's evidence unchanged. Some historical
CSV dependencies are copied/hashed as opaque files; neither child imports the raw
market loader or parses the final market values.

This replay repeats saved-data postprocessing and the prespecified static
references. It does not refit, reselect or rerun the main model or its final
evaluation. These new diagnostics currently require the full saved-input
repository; the earlier compact V2 ZIP continues to reproduce its original fixed
model results. Integrating the new supporting source/evidence into the final
submission package is a separate delivery step when the paper is prepared.

AI assistance, source attribution and the synthetic-data limitations continue to
apply as recorded in `AI_USE.md` and `../docs/DATA_NOTICE.md`.
