# Migration map

| Original location | Current location | Reason |
| --- | --- | --- |
| Root engine/config/exact mathematics | Same paths | Preserve algorithm and numerical behavior. |
| `Improvements/research/` | `research/` | One importable research stack aligned with root engine. |
| `Improvements/strategies/adaptive_final.py` | `strategies/adaptive_standalone.py` | Baseline used by historical Phase 1, no local imports, recommended CLI bot. |
| Root `strategies/adaptive_final.py` | Same path | Modular sweep-derived strategy remains available for programmatic evaluation. |
| `Improvements/adaptive_final(2).py` | `strategies/competition_base.py` | Descriptive builder input; original also archived. |
| `Improvements/strategies/adaptive_competition.py` | `strategies/adaptive_competition.py` | Existing table-embedded research candidate. |
| `Improvements/build_competition_bot.py` | `build_competition_bot.py` | Portable default input/output paths. |
| Remaining `Improvements/` | `experiments/archive/improvements/` | Preserve historical variants, including duplicate engine snapshots. |
| `test/quantstorm-ps/` | `experiments/archive/competition_snapshot/` | Distinct engine and upstream checkout; separate from current tests. |
| Nested loader/policy/sandbox/limits/starter/rulebook | Root copies | Supply dependencies missing from root backtester and document rules. |
| Original `__pycache__`, Git metadata, large data | Private backup / ignored local files | Preserve originals without publishing generated or sensitive artifacts. |

All original files were backed up before migration in
`.reconstruction/original-project.zip`; SHA-256 inventory is
`.reconstruction/inventory.csv`. The detailed pre-edit report is
`.reconstruction/reconstruction-report.json`; nested refs and history are also
saved as `.reconstruction/upstream-history.bundle`. These remain uncommitted.

Behavior changes: the CLI defaults to standalone AdaptiveFinal versus Rational,
rejects nonpositive deal counts, and resolves default bot paths from the project
directory. No engine rules or EV algorithms were changed. The Phase 1 baseline
now explicitly uses the original standalone implementation and writes a new
report instead of overwriting evidence. Metadata headers are anonymized.
