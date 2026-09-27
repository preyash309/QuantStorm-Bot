"""
PHASE 4B — One-step opponent-conditioned negotiation EV.
"""

from __future__ import annotations

from dataclasses import dataclass

from .negotiation_ev import (
    NegotiationAction,
    accept_buy_ev,
    accept_sell_ev,
    enumerate_actions,
)


@dataclass(frozen=True)
class ContinuationEV:

    action: NegotiationAction

    ev: float

    p_accept_buy: float

    p_accept_sell: float

    p_counter: float


def evaluate_counter(
    *,
    obs,
    current_quote,
    candidate,
    opponent_model,
    next_turn,
    final_cap,
    min_reduction,
    forcing_fee=2.0,
    forced_shift=0,
):
    """
    Evaluate a candidate counter using one-step opponent
    response probabilities.

    Model:

        EV(C) =
            P(B) * EV(B)
          + P(S) * EV(S)
          + P(C) * terminal_EV

    The continuation after the opponent counters is currently
    approximated by the forced-fill value of the candidate
    range.

    Phase 4C will replace this terminal approximation with
    backward induction.
    """

    if candidate.kind != "COUNTER":
        raise ValueError(
            "candidate must be COUNTER"
        )

    new_quote = (
        candidate.bid,
        candidate.ask,
    )

    # --------------------------------------------------------
    # What the opponent sees after our counter.
    # --------------------------------------------------------

    probabilities = opponent_model.predict(
        round_no=int(obs.round),

        # We need the opponent's perspective.
        #
        # Our te_mine becomes their te_theirs.
        # Our te_theirs becomes their te_mine.
        #
        # Our powers_mine become their powers_theirs.
        # Our powers_theirs become their powers_mine.
        is_maker=not bool(
            obs.is_maker
        ),

        te_mine=int(
            obs.te_theirs
        ),

        te_theirs=int(
            obs.te_mine
        ),

        powers_mine=obs.powers_theirs,

        powers_theirs=obs.powers_mine,

        quote=new_quote,

        turn=next_turn,
    )

    p_buy = probabilities[
        "ACCEPT_BUY"
    ]

    p_sell = probabilities[
        "ACCEPT_SELL"
    ]

    p_counter = probabilities[
        "COUNTER"
    ]

    # --------------------------------------------------------
    # If opponent accepts BUY:
    #
    # Opponent buys at ask.
    # We are SHORT at ask.
    #
    # Our EV = ask - E[S]
    #
    # --------------------------------------------------------

    ev_if_buy = accept_sell_ev(
        obs,
        new_quote[1],
    )

    # --------------------------------------------------------
    # If opponent accepts SELL:
    #
    # Opponent sells at bid.
    # We are LONG at bid.
    #
    # --------------------------------------------------------

    ev_if_sell = accept_buy_ev(
        obs,
        new_quote[0],
    )

    # --------------------------------------------------------
    # If opponent counters:
    #
    # One-step baseline:
    # assume the current range eventually reaches forced fill.
    #
    # This is deliberately conservative and will be replaced
    # by Phase 4C.
    # --------------------------------------------------------

    midpoint = (
        new_quote[0]
        + new_quote[1]
    ) // 2

    forced_price = (
        midpoint
        + forced_shift
    )

    ev_if_counter = (
        forced_price
        - (
            float(obs.k_mine)
            + float(obs.foresight_sum)
        )
        - forcing_fee
    )

    ev = (
        p_buy * ev_if_buy
        +
        p_sell * ev_if_sell
        +
        p_counter * ev_if_counter
    )

    return ContinuationEV(
        action=candidate,
        ev=ev,
        p_accept_buy=p_buy,
        p_accept_sell=p_sell,
        p_counter=p_counter,
    )


def evaluate_all_actions(
    *,
    obs,
    current_quote,
    opponent_model,
    turn,
    final_cap,
    min_reduction,
    forcing_fee=2.0,
    forced_shift=0,
):
    """
    Evaluate ACCEPT_BUY, ACCEPT_SELL and every legal COUNTER.
    """

    actions = enumerate_actions(
        bid=current_quote[0],
        ask=current_quote[1],
        final_cap=final_cap,
        min_reduction=min_reduction,
    )

    results = []

    for action in actions:

        if action.kind == "ACCEPT_BUY":

            ev = accept_buy_ev(
                obs,
                current_quote[1],
            )

            results.append(
                ContinuationEV(
                    action=action,
                    ev=ev,
                    p_accept_buy=1.0,
                    p_accept_sell=0.0,
                    p_counter=0.0,
                )
            )

            continue

        if action.kind == "ACCEPT_SELL":

            ev = accept_sell_ev(
                obs,
                current_quote[0],
            )

            results.append(
                ContinuationEV(
                    action=action,
                    ev=ev,
                    p_accept_buy=0.0,
                    p_accept_sell=1.0,
                    p_counter=0.0,
                )
            )

            continue

        result = evaluate_counter(
            obs=obs,
            current_quote=current_quote,
            candidate=action,
            opponent_model=opponent_model,
            next_turn=turn + 1,
            final_cap=final_cap,
            min_reduction=min_reduction,
            forcing_fee=forcing_fee,
            forced_shift=forced_shift,
        )

        results.append(result)

    return tuple(
        sorted(
            results,
            key=lambda x: x.ev,
            reverse=True,
        )
    )