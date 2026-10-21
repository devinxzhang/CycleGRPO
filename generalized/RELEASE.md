# Video code release — 2026-10-21

## Included

- Isolated `generalized.verl` runtime; the existing ECCV `verl` is unchanged.
- Qwen and OMTG temporal recipes, and the main mixed Open-o3 recipe.
- Nested caption/localization rollouts, temporal source routing, sparse
  spatio-temporal overlap, query-conditioning, optional video frame cache,
  token-level policy optimization, and inference checkpoint merging.
- Public-path versions of the Open-o3 converters, audits, QA mixing, and
  transparent temporal-hard sampling utilities.
- Input preflight, configurable foreground launchers, and CPU regression tests.

## Validation

The release passed 27 CPU tests; one optional real-video fixture test was
skipped. All added Python files passed syntax compilation, the Bash launcher
passed `bash -n`, and the five main data-construction tools expose working
CLI help. Launcher dry-runs were checked for all three profiles with synthetic
local checkpoint/data inventories. No GPU optimization step was executed
as part of this packaging update.

Checkpoint inventory/header checks were performed privately against the
three retained paper models: the Qwen-initialized temporal model, the
OMTG-initialized temporal model, and the Open-o3-initialized spatio-temporal
model. Public model names identify their capabilities, not internal training
identifiers. The checks do not republish private paths or re-run benchmarks.

All newly released files were checked for credential patterns, internal
service addresses, and private filesystem paths. Original third-party
copyright and license notices are retained.
