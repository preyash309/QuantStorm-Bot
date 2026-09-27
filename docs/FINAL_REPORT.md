# Reconstruction and publication report

## Project summary

Divided Oracle is a standard-library Python trading-game strategy project with
exact probability mathematics, adaptive auctions, independent simulation and
empirical opponent-policy research. It does not require GPUs or ML checkpoints.

## Structure and changes

Consolidated the duplicated research stack into `research/`; restored missing
loader/policy/sandbox dependencies; established the reproduced standalone bot
as the default CLI path; preserved modular alternatives and incompatible
competition snapshots under an explicit archive. Core numerical source files
and game rules were verified unchanged. Default CLI paths are portable and
nonpositive deal counts produce a clear error.

## Preserved research

Retained adaptive/exact/denial/shading strategy progression, independent replay
mechanics, Phase 1 baseline, Phase 3 state/calibration reports, Phase 4
negotiation studies, action audits, negative policy-sanity findings, and the
table-embedding builder. Large raw datasets remain local. Original files,
untracked work and nested upstream Git history have a private recoverable backup.

## Verification

- All 14 test programs pass locally and from a clean Git archive.
- Three original baseline PnL totals reproduced exactly over 600 deals.
- Embedded competition candidate and isolated standalone matches passed smoke checks.
- GitHub Actions Windows/Linux Python 3.11/3.14 workflow completed successfully.
- No broken maintained Markdown links, Git submodule links or excluded binary
  artifacts were found in the 155-file publication tree before this report.

## Limitations

Full parameter sweeps and historical held-out policy experiments were not rerun.
Some original sweep conclusions have no retained raw logs. Regeneration defaults
do not necessarily match historical dataset seeds. The isolation smoke test
does not prove general security. No explicit upstream license was found;
no new redistribution license has been granted.

## Publication exclusions

Ignored private reconstruction evidence/backups, original nested Git metadata,
caches/virtual environments/env files, three large observation datasets,
generated policy exports, sponsor images and old promotional README. Real
participant metadata was anonymized in repository copies. Pattern scans found
no recognized credential material; this does not guarantee absence of secrets.

## GitHub status

Published as a **private** repository at
[preyash309/quantstorm-divided-oracle-research](https://github.com/preyash309/quantstorm-divided-oracle-research).
Default branch: `main`. Source, testing, and documentation changes were committed
separately. Remote visibility, default branch and commit equality were verified
through the authenticated GitHub API. The final documentation commit is
reported in the task's completion message and available with `git rev-parse HEAD`.
There are no remaining publication blockers.
