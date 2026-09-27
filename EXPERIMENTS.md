# Experiments and retained evidence

The supported baseline is the standalone AdaptiveFinal bot. Filenames and
comments alone do not establish performance. The table separates retained
measurements from implementation claims.

| Study | Code / evidence | Purpose and status |
| --- | --- | --- |
| Reference pricing | `strategies/naive_ev.py`, `strategies/rational.py` | Reference opponent policies used by collector and reproduced baseline. |
| Adaptive progression | `adaptive_bidder.py`, `adaptive_plus.py`, `adaptive_exact.py` | Quote inference, auction decisions, exact final-turn EV; kept as meaningful alternatives. |
| TRANSFORM denial | `adaptive_exact_denial.py`, `denial_sweep.py` | Values purchasing a swap versus denying it; original large-sweep conclusion is not independently reproduced. |
| Auction shade | `adaptive_power.py`, `shade_sweep.py`, `transform_shade_sweep.py` | Parameter sweeps; frozen shade 0.60 and denial weight 1.10 are encoded in AdaptiveFinal. No raw sweep logs were retained. |
| Robustness | `final_robustness.py`, `research_harness.py` | Large fresh-seed comparison (50 matches x 1,000 direct deals per opponent, mirrored); entry point retained but not rerun. |
| Phase 1 | `research/baseline/run_baseline.py`, `research/results/phase1_baseline.json` | 200 deals per opponent; all three PnL totals independently reproduced. |
| Phase 2 | `research/simulator/`, `research/tests/test_trace_replay.py` | Independent mechanics; test compares score, TE, contracts and PnL to engine on recorded actions. Passed. |
| Phase 3 | `research/opponent/` | Smoothed conditional action frequencies, coarse/exact/engineered state comparisons and calibration. Seed-based held-out reports retained; full collection not rerun. |
| Phase 4B/C | `research/ev/public_opponent.py`, `quote_opponent.py`, `phase4b.py`, `phase4c.py` | Public-state and quote-conditioned responses; small unit fixtures passed. Historical validation reports retained. |
| Phase 4D | `research/ev/phase4d.py`, `negotiation_ev.py`, `continuation.py` | One-step opponent-conditioned EV; continuation is an approximation, not a fully solved game. |
| Phase 4D.1 | `research/results/phase4d1_action_audit.json` | Historical audit: 26,951/26,951 actions matched, zero skipped; not rerun. |
| Phase 4E | `research/ev/phase4e0_counter_policy.py`, `phase4e1_policy_sanity.py` | Exact counter-action policy and sanity check; historical accuracy 0.765612, log loss 0.883576, unseen rate 0.147082. Predicted COUNTER on 26,936 of 26,951 rows; evidence of overconfidence. |
| Competition embedding | `build_competition_bot.py`, `strategies/adaptive_competition.py` | Embeds empirical tables, overrides pre-final actions only above an EV gain gate of 0.35. Four-deal smoke test passed; superiority not established. |

Reports live in `research/results/`. Large datasets, empirical model exports,
and generated outputs remain local and ignored. Phase scripts generally accept
`--input`; consult their `--help` and run from the root. The original
seed-based splits and model features are unchanged. Opponent features exclude
the true hidden score and the private opposing hand.

## Archival implementations

`experiments/archive/improvements/` preserves the original standalone
development/builder variants. Its separate research tree was consolidated into
`research/`. `experiments/archive/competition_snapshot/` retains a different
competition engine and test1..test6/testc1..testc6 variants. Those filenames
are historical parameter variants, not automated test suite members. No
retained paired evidence establishes a winner among them.

Archive entry points are unsupported and may retain assumptions about their
original working directories. The original private ZIP preserves their exact
contents, including identifiers that were sanitized in repository copies.
