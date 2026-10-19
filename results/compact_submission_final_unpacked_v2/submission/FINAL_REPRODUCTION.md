# Fixed final-table reproduction in technical source V2

V2 adds `scripts/submission_final_adapter.py`; the V1 ZIP, old adapters, original
producer, kernels and numerical archives remain unchanged. Install the pinned
research dependencies and run this new entrypoint without a flag for a complete
14-object smoke run using constructed all-zero market/inflation arrays:

```sh
python scripts/submission_final_adapter.py
```

This default runs the actual frozen execution kernels and pure table helpers,
with 120 constructed training paths and 60 constructed evaluation paths. It
does not parse the supplied market CSV. IDs 121–180 in this mode are synthetic
labels only; the receipt marks `market_is_constructed=true` and
`final_scenarios_loaded=false`.

Only the following explicit command may reproduce the fixed final tables:

```sh
python scripts/submission_final_adapter.py --reproduce-fixed-nonblind-final
```

It first verifies the complete compact source map, locked registry and object
specification, final-addon manifest and canonical expectation release. It then
loads training IDs 1–120 and evaluation IDs 121–180 with the declared separate
loader gates. It reproduces all 14 objects: the Q2 main model and three
references; Q1 own-horizon A–E with A/D Review financial fields NULL; and five
own-horizon exact-thirds references. It checks every saved array field's
dtype/shape/value digest plus the byte identity of all ten table/JSON artifacts,
including 14 summaries, eight paired comparisons and the original severe-case
rule. NPZ container bytes need not match because array contents are checked.

These final scenarios had already been exposed during the project. Repeating
them is fixed-table reproduction, not a new independent validation sample.
No fitting, candidate selection, parameter change, new case selection rule or
historical gate removal occurs. The original producer's `run` and its large
historical evidence gates are not invoked or replaced by this compact adapter.

Each command preserves its complete, failed or incomplete attempt and refuses
to overwrite it. Use `--output <new-directory>` for an explicitly intended
separate run. Review the AI-use and dataset attribution notices included with
V1; no additional code licence or external peer-review claim is introduced.
