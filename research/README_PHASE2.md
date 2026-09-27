# Phase 2 — Independent Exact Simulator

The official `engine.py` remains the execution oracle. The files in
`research/simulator/` implement the game mechanics independently.

Run from the QuantStorm project root:

```powershell
python research\tests\test_score.py
python research\tests\test_mechanics.py
python research\tests\test_simulator_smoke.py
python research\tests\test_trace_replay.py
```

The critical test is `test_trace_replay.py`.

It:
1. executes a real deal through the official engine;
2. records the exact bot actions using the Phase-1 TraceBot;
3. converts those actions into explicit research actions;
4. runs the same actions through the independent simulator;
5. compares score, TE, every contract field, and final PnL.

Do not start dynamic programming until this parity test passes.

The simulator is intentionally policy-agnostic. It does not decide actions yet.
Phase 3 will add opponent policies and backward induction.
