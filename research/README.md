# Research layer

Run from the repository root. `engine.py` is the execution oracle;
`simulator/` independently models state, actions, mechanics, scoring and replay.
`opponent/` collects information-safe observations and fits smoothed action
frequency models. `ev/` contains public/quote-conditioned policies, action
enumeration, and one-step negotiation continuation experiments.

```bash
python -B tools/run_tests.py
python -B research/baseline/run_baseline.py
python -B research/opponent/run_phase3.py
```

The first two commands were verified during reconstruction. Full trajectory
collection was not rerun. Most research commands write generated files under
`results/`; the baseline uses a separate reproduced report. Historical small
reports are committed, while large row datasets and policy exports are ignored.

Read [experiment history](../EXPERIMENTS.md) and the historical
[Phase 2](README_PHASE2.md) / [Phase 3](README_PHASE3.md) notes for methodology.
