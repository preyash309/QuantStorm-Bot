# Reconstruction verification

Local environment: Windows, CPython 3.14.7, standard library only.

## Tests executed

`python -B tools/run_tests.py`: **14/14 test programs passed**.

- Two original exact-probability/final-turn-EV programs.
- Seven original research programs covering immutable state, engine adapter
  parity, scoring, mechanics, simulator smoke, trace replay and opponent policy.
- Four original negotiation/public/quote-conditioned EV test programs.
- One new runtime unittest program with five regressions: deterministic
  mirrored zero-sum matches, restored loader, CLI startup from another working
  directory, nonpositive deal rejection, and frozen rules.

The initial negotiation test failed because it imposed width >= final_cap.
The implementation and root engine sanitizer both accept widths down to zero.
The assertion was corrected and a zero-width enumeration check was added.
No strategy mathematics or game mechanics were changed.

## Match checks

- Phase 1 baseline: 600 total deals (200 per opponent), seed 900001.
  All historical PnL totals reproduced exactly. No warnings, violations,
  clamps or forfeits. Fresh report: `research/results/phase1_baseline_reproduced.json`.
- Embedded competition candidate versus Rational: 4 mirrored deals, seed 42,
  total PnL approximately +2.05, no warnings. This is a startup smoke test,
  not a performance comparison with statistical power.
- Default standalone versus Rational under `--isolate`: 2 mirrored deals,
  seed 42, aggregate PnL approximately +4.52, no warnings. The workspace
  sandbox initially denied the Windows multiprocessing pipe; the same command
  succeeded when executed with approved elevated tool permissions.
- Static `--validate` accepted the standalone example. The gate accepts its
  placeholder metadata, so acceptance does not prove participant identity.

## Security and history

Before migration, all originals were backed up with a SHA-256 file inventory.
The nested upstream history is saved as a Git bundle. Pattern scans inspected
29 distinct text blobs from all nested refs, as well as current text files.
No GitHub token, AWS access key, private-key block or suspicious credential
assignment was found. Some historical metadata examples matched the identifier
scan; current real participant headers and a personal email were anonymized.
The full original history is retained locally and is not imported into the new
publication branch. Pattern scanning is not a proof of absence of secrets.

Excluded: the private backup/report/history bundle, nested `.git`, caches,
virtual environments, env files, large datasets/policy exports, generated
outputs, sponsor images and the promotional upstream README. Small scientific
reports and the freshly reproduced baseline remain in Git.

## Validation limits

Full parameter sweeps, trajectory regeneration, model rebuilding, and all
historical held-out policy reports were not rerun. Dataset split logic and
algorithms were preserved. CI defines Linux/Windows Python 3.11/3.14 checks;
those remote jobs are pending publication. The sandbox smoke check is limited
and does not establish resistance to arbitrary hostile Python programs.
No explicit upstream license was found; no new license has been granted.
