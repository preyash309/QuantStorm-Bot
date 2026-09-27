# Phase 3 — Empirical Opponent Policy

## Goal

Measure how the actual reference bots behave, using the official engine as
the oracle.

This phase does NOT optimize our bot and does NOT modify `adaptive_final.py`.

The model is:

    P(opponent action | information available to opponent)

It is a transparent conditional-frequency model with Laplace smoothing.

## Run

From:

    the repository root

first run:

    python research\tests\test_opponent_policy.py

Then:

    python research\opponent\run_phase3.py

Then:

    python research\opponent\analyze.py research\results\phase3_policy_dataset.json

## Dataset

The collector records only:

- round
- k_mine
- TE
- maker/taker status
- active powers
- FORESIGHT information
- public auction log
- offered powers
- quote
- turn
- action

It deliberately does NOT record the true hidden score or private opponent
hand as model features.

The official engine is used for every trajectory. Its documented deal
construction uses deterministic seed streams, mirrored hands/roles, and a
separate symmetric stream for tie/FORESIGHT randomness.

## Why this is Phase 3

We now have:

Phase 1: information-safe state representation.
Phase 2: independent game mechanics with trace parity.
Phase 3: empirical opponent behaviour.

The output is the first usable approximation to:

    P(a_opp | s)

which Phase 4 will use in backward induction / negotiation EV.

## Important

Do not put the generated JSON into the final submission. It is research
data only.

Do not train a neural network here. The state space is discrete and the
reference bots are deterministic policies. Exact conditional counts are
more interpretable and easier to validate.
