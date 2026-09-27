"""
BUILD COMPETITION BOT
=====================

Builds a standalone competition bot from:

    adaptive_final(2).py
    research/opponent/phase3_policy_dataset.json

Research integrated:

    Phase 4C:
        P(opponent response | public state, quote, turn)

    Phase 4D:
        one-step opponent-conditioned negotiation EV

The generated bot:

    - does NOT need the 90 MB dataset at runtime
    - does NOT import research.*
    - uses only tournament-safe Python constructs
    - keeps the existing final-turn optimizer unchanged
    - uses the research negotiation layer only on pre-final turns
    - falls back to the original Adaptive policy when the
      research model does not provide a sufficiently strong
      improvement

Output:

    adaptive_final_competition.py
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# Constants
# ============================================================

ACTIONS = (
    "ACCEPT_BUY",
    "ACCEPT_SELL",
    "COUNTER",
)

# Conservative deployment gate.
#
# The research action must beat the existing Adaptive action
# by at least this much estimated EV before we override it.
RESEARCH_MIN_EV_GAIN = 0.35


# ============================================================
# Dataset
# ============================================================

def load_rows(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            "Dataset root must be a JSON object."
        )

    if "rows" not in data:
        raise RuntimeError(
            "Dataset does not contain top-level 'rows'."
        )

    rows = data["rows"]

    if not isinstance(
        rows,
        list,
    ):
        raise RuntimeError(
            "'rows' must be a JSON array."
        )

    return rows


# ============================================================
# EXACT PHASE-4C STATE KEY
# ============================================================

def quote_state_key(row):

    """
    Mirrors the Phase-4C quote-conditioned state definition.
    """

    obs = row["obs"]

    quote = row.get(
        "quote"
    )

    if quote is None:

        quote = (
            None,
            None,
        )

    else:

        quote = (
            int(quote[0]),
            int(quote[1]),
        )

    powers_mine = tuple(
        sorted(
            str(x)
            for x in obs.get(
                "powers_mine",
                [],
            )
        )
    )

    powers_theirs = tuple(
        sorted(
            str(x)
            for x in obs.get(
                "powers_theirs",
                [],
            )
        )
    )

    return (
        int(obs["round"]),
        int(obs["te_mine"]),
        int(obs["te_theirs"]),
        bool(obs["is_maker"]),
        powers_mine,
        powers_theirs,
        int(row.get("turn", 0)),
        quote,
    )


# ============================================================
# FIT PHASE-4C MODEL
# ============================================================

def fit_quote_model(rows):

    """
    Fit the empirical opponent response policy.

    We use ALL response observations for the final deployment
    model. The held-out validation work has already been performed
    separately.
    """

    tables = defaultdict(
        Counter
    )

    global_counts = Counter()

    n_rows = 0

    for row in rows:

        if row.get(
            "method"
        ) != "respond":

            continue

        action = row.get(
            "action"
        )

        if isinstance(
            action,
            list,
        ):

            if not action:
                continue

            action_name = str(
                action[0]
            )

        else:

            action_name = str(
                action
            )

        if action_name not in ACTIONS:
            continue

        key = quote_state_key(
            row
        )

        tables[key][
            action_name
        ] += 1

        global_counts[
            action_name
        ] += 1

        n_rows += 1

    return (
        tables,
        global_counts,
        n_rows,
    )


# ============================================================
# SERIALIZE MODEL AS NATIVE PYTHON
# ============================================================

def serialize_model(
    tables,
    global_counts,
):

    """
    Serialize the Phase-4C model as a native Python literal.

    IMPORTANT:

    We intentionally do NOT use:

        base64
        json
        zlib

    in the generated competition bot.

    The tournament sandbox only permits a small standard-library
    whitelist, so the model is embedded directly as Python data.
    """

    states = []

    for key, counts in tables.items():

        (
            round_no,
            te_mine,
            te_theirs,
            is_maker,
            powers_mine,
            powers_theirs,
            turn,
            quote,
        ) = key

        quote_bid, quote_ask = quote

        states.append(
            (
                int(round_no),
                int(te_mine),
                int(te_theirs),
                bool(is_maker),

                tuple(
                    powers_mine
                ),

                tuple(
                    powers_theirs
                ),

                int(turn),

                quote_bid,
                quote_ask,

                int(
                    counts[
                        "ACCEPT_BUY"
                    ]
                ),

                int(
                    counts[
                        "ACCEPT_SELL"
                    ]
                ),

                int(
                    counts[
                        "COUNTER"
                    ]
                ),
            )
        )

    payload = {
        "states": tuple(
            states
        ),

        "global": (
            int(
                global_counts[
                    "ACCEPT_BUY"
                ]
            ),

            int(
                global_counts[
                    "ACCEPT_SELL"
                ]
            ),

            int(
                global_counts[
                    "COUNTER"
                ]
            ),
        ),
    }

    return repr(
        payload
    )


# ============================================================
# RUNTIME CODE
# ============================================================

RUNTIME_CODE = r'''
# ============================================================
# EMBEDDED PHASE-4C / PHASE-4D RESEARCH LAYER
# ============================================================

_QT_MODEL_DATA = __MODEL_DATA__


# ------------------------------------------------------------
# Load embedded empirical policy
# ------------------------------------------------------------

def _qt_load_model():

    payload = _QT_MODEL_DATA

    tables = {}

    for state in payload["states"]:

        (
            round_no,
            te_mine,
            te_theirs,
            is_maker,
            powers_mine,
            powers_theirs,
            turn,
            quote_bid,
            quote_ask,
            n_buy,
            n_sell,
            n_counter,
        ) = state

        key = (
            int(round_no),

            int(te_mine),

            int(te_theirs),

            bool(
                is_maker
            ),

            tuple(
                sorted(
                    str(x)
                    for x in powers_mine
                )
            ),

            tuple(
                sorted(
                    str(x)
                    for x in powers_theirs
                )
            ),

            int(turn),

            (
                None
                if quote_bid is None
                else int(quote_bid),

                None
                if quote_ask is None
                else int(quote_ask),
            ),
        )

        tables[key] = (
            int(n_buy),
            int(n_sell),
            int(n_counter),
        )

    global_counts = tuple(
        int(x)
        for x in payload["global"]
    )

    return (
        tables,
        global_counts,
    )


_QT_TABLES, _QT_GLOBAL = (
    _qt_load_model()
)


# ------------------------------------------------------------
# Exact Phase-4C key
# ------------------------------------------------------------

def _qt_state_key(
    *,
    round_no,
    te_mine,
    te_theirs,
    is_maker,
    powers_mine,
    powers_theirs,
    turn,
    quote,
):

    if quote is None:

        q = (
            None,
            None,
        )

    else:

        q = (
            int(
                quote[0]
            ),

            int(
                quote[1]
            ),
        )

    return (
        int(round_no),

        int(te_mine),

        int(te_theirs),

        bool(
            is_maker
        ),

        tuple(
            sorted(
                str(x)
                for x in powers_mine
            )
        ),

        tuple(
            sorted(
                str(x)
                for x in powers_theirs
            )
        ),

        int(turn),

        q,
    )


# ------------------------------------------------------------
# Probability prediction
# ------------------------------------------------------------

def _qt_predict(
    *,
    round_no,
    te_mine,
    te_theirs,
    is_maker,
    powers_mine,
    powers_theirs,
    turn,
    quote,
    smoothing=1.0,
):

    key = _qt_state_key(
        round_no=round_no,
        te_mine=te_mine,
        te_theirs=te_theirs,
        is_maker=is_maker,
        powers_mine=powers_mine,
        powers_theirs=powers_theirs,
        turn=turn,
        quote=quote,
    )

    counts = _QT_TABLES.get(
        key
    )

    if counts is None:

        counts = _QT_GLOBAL

    total = sum(
        counts
    )

    denominator = (
        total
        + smoothing * 3.0
    )

    return {
        "ACCEPT_BUY":
            (
                counts[0]
                + smoothing
            )
            / denominator,

        "ACCEPT_SELL":
            (
                counts[1]
                + smoothing
            )
            / denominator,

        "COUNTER":
            (
                counts[2]
                + smoothing
            )
            / denominator,
    }


# ------------------------------------------------------------
# State availability
# ------------------------------------------------------------

def _qt_known_state(
    *,
    round_no,
    te_mine,
    te_theirs,
    is_maker,
    powers_mine,
    powers_theirs,
    turn,
    quote,
):

    key = _qt_state_key(
        round_no=round_no,
        te_mine=te_mine,
        te_theirs=te_theirs,
        is_maker=is_maker,
        powers_mine=powers_mine,
        powers_theirs=powers_theirs,
        turn=turn,
        quote=quote,
    )

    return (
        key
        in _QT_TABLES
    )


# ------------------------------------------------------------
# Exact value
# ------------------------------------------------------------

def _qt_posterior_mean(
    self,
    obs,
):

    """
    Use the existing AdaptiveFinal valuation.

    We intentionally avoid creating a second valuation model.
    """

    try:

        return float(
            self._exact_value(
                obs
            )
        )

    except Exception:

        value = float(
            obs.k_mine
        )

        if hasattr(
            obs,
            "foresight",
        ):

            try:

                value += float(
                    sum(
                        obs.foresight
                    )
                )

            except Exception:

                pass

        return value


# ------------------------------------------------------------
# Legal counter enumeration
# ------------------------------------------------------------

def _qt_enumerate_counters(
    *,
    bid,
    ask,
    final_cap,
    min_reduction,
):

    if bid > ask:

        bid, ask = (
            ask,
            bid,
        )

    current_width = (
        ask - bid
    )

    max_width = min(
        current_width,

        max(
            final_cap,

            current_width
            - min_reduction,
        ),
    )

    counters = []

    for new_bid in range(
        bid,
        ask + 1,
    ):

        for new_ask in range(
            new_bid,
            ask + 1,
        ):

            width = (
                new_ask
                - new_bid
            )

            if width > max_width:
                continue

            counters.append(
                (
                    "COUNTER",
                    new_bid,
                    new_ask,
                )
            )

    counters.sort(
        key=lambda action: (
            action[2]
            - action[1],

            (
                action[1]
                + action[2]
            )
            / 2.0,

            action[1],

            action[2],
        )
    )

    return counters


# ------------------------------------------------------------
# Direct EV
# ------------------------------------------------------------

def _qt_accept_buy_ev(
    mean,
    ask,
):

    return (
        mean
        - ask
    )


def _qt_accept_sell_ev(
    mean,
    bid,
):

    return (
        bid
        - mean
    )


# ------------------------------------------------------------
# Counter EV
# ------------------------------------------------------------

def _qt_counter_ev(
    self,
    obs,
    candidate,
    turn,
):

    new_bid = int(
        candidate[1]
    )

    new_ask = int(
        candidate[2]
    )

    next_turn = int(
        turn + 1
    )

    # --------------------------------------------------------
    # Opponent perspective.
    #
    # Our:
    #
    #   te_mine       -> opponent te_theirs
    #   te_theirs     -> opponent te_mine
    #   powers_mine   -> opponent powers_theirs
    #   powers_theirs -> opponent powers_mine
    #
    # --------------------------------------------------------

    known = _qt_known_state(
        round_no=int(
            obs.round
        ),

        te_mine=int(
            obs.te_theirs
        ),

        te_theirs=int(
            obs.te_mine
        ),

        is_maker=not bool(
            obs.is_maker
        ),

        powers_mine=(
            obs.powers_theirs
        ),

        powers_theirs=(
            obs.powers_mine
        ),

        turn=next_turn,

        quote=(
            new_bid,
            new_ask,
        ),
    )

    probabilities = _qt_predict(
        round_no=int(
            obs.round
        ),

        te_mine=int(
            obs.te_theirs
        ),

        te_theirs=int(
            obs.te_mine
        ),

        is_maker=not bool(
            obs.is_maker
        ),

        powers_mine=(
            obs.powers_theirs
        ),

        powers_theirs=(
            obs.powers_mine
        ),

        turn=next_turn,

        quote=(
            new_bid,
            new_ask,
        ),
    )

    mean = _qt_posterior_mean(
        self,
        obs,
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
    # Opponent accepts BUY.
    #
    # Opponent buys at our ask.
    # We are short at ask.
    # --------------------------------------------------------

    ev_if_buy = (
        new_ask
        - mean
    )

    # --------------------------------------------------------
    # Opponent accepts SELL.
    #
    # Opponent sells at our bid.
    # We are long at bid.
    # --------------------------------------------------------

    ev_if_sell = (
        mean
        - new_bid
    )

    # --------------------------------------------------------
    # Opponent counters.
    #
    # One-step forced-fill approximation from Phase 4D.
    # --------------------------------------------------------

    midpoint = (
        new_bid
        + new_ask
    ) // 2

    forcing_fee = 2.0

    ev_if_counter = (
        midpoint
        - mean
        - forcing_fee
    )

    ev = (
        p_buy
        * ev_if_buy

        +

        p_sell
        * ev_if_sell

        +

        p_counter
        * ev_if_counter
    )

    return (
        float(ev),
        probabilities,
        known,
    )


# ------------------------------------------------------------
# Normalize Adaptive action
# ------------------------------------------------------------

def _qt_normalize_action(
    action,
):

    if isinstance(
        action,
        str,
    ):

        if action in (
            "ACCEPT_BUY",
            "ACCEPT_SELL",
        ):

            return (
                action,
                None,
                None,
            )

    if isinstance(
        action,
        tuple,
    ):

        if (
            len(action) >= 3
            and action[0]
            == "COUNTER"
        ):

            return (
                "COUNTER",
                int(
                    action[1]
                ),
                int(
                    action[2]
                ),
            )

    if isinstance(
        action,
        list,
    ):

        if (
            len(action) >= 3
            and action[0]
            == "COUNTER"
        ):

            return (
                "COUNTER",
                int(
                    action[1]
                ),
                int(
                    action[2]
                ),
            )

    return None


# ------------------------------------------------------------
# Baseline EV
# ------------------------------------------------------------

def _qt_baseline_ev(
    self,
    obs,
    current_quote,
    action,
    turn,
):

    normalized = (
        _qt_normalize_action(
            action
        )
    )

    if normalized is None:
        return None

    kind, bid, ask = (
        normalized
    )

    mean = _qt_posterior_mean(
        self,
        obs,
    )

    if kind == "ACCEPT_BUY":

        return (
            _qt_accept_buy_ev(
                mean,
                current_quote[1],
            )
        )

    if kind == "ACCEPT_SELL":

        return (
            _qt_accept_sell_ev(
                mean,
                current_quote[0],
            )
        )

    if kind == "COUNTER":

        result = _qt_counter_ev(
            self,
            obs,
            normalized,
            turn,
        )

        return result[0]

    return None


# ------------------------------------------------------------
# Main research selector
# ------------------------------------------------------------

def _qt_research_response(
    self,
    obs,
    quote,
    turn,
):

    # Never interfere with exact final-turn optimization.
    if (
        turn
        == self.config.N_TURNS
    ):

        return None

    bid = int(
        quote[0]
    )

    ask = int(
        quote[1]
    )

    # --------------------------------------------------------
    # Existing Adaptive action.
    # --------------------------------------------------------

    baseline_action = (
        self._normal_response(
            obs,
            quote,
        )
    )

    baseline_ev = (
        _qt_baseline_ev(
            self,
            obs,
            quote,
            baseline_action,
            turn,
        )
    )

    if baseline_ev is None:
        return None

    # --------------------------------------------------------
    # Candidate actions.
    # --------------------------------------------------------

    candidates = [
        (
            "ACCEPT_BUY",
            None,
            None,
        ),

        (
            "ACCEPT_SELL",
            None,
            None,
        ),
    ]

    final_cap = int(
        obs.final_cap
    )

    min_reduction = int(
        self.config.MIN_REDUCTION
    )

    candidates.extend(
        _qt_enumerate_counters(
            bid=bid,
            ask=ask,
            final_cap=final_cap,
            min_reduction=min_reduction,
        )
    )

    # --------------------------------------------------------
    # Search.
    # --------------------------------------------------------

    best_action = None
    best_ev = None

    for candidate in candidates:

        kind = candidate[0]

        # ----------------------------------------------------
        # ACCEPT_BUY
        # ----------------------------------------------------

        if kind == "ACCEPT_BUY":

            mean = _qt_posterior_mean(
                self,
                obs,
            )

            ev = (
                _qt_accept_buy_ev(
                    mean,
                    ask,
                )
            )

            if (
                best_ev is None
                or ev > best_ev
            ):

                best_ev = ev
                best_action = candidate

            continue

        # ----------------------------------------------------
        # ACCEPT_SELL
        # ----------------------------------------------------

        if kind == "ACCEPT_SELL":

            mean = _qt_posterior_mean(
                self,
                obs,
            )

            ev = (
                _qt_accept_sell_ev(
                    mean,
                    bid,
                )
            )

            if (
                best_ev is None
                or ev > best_ev
            ):

                best_ev = ev
                best_action = candidate

            continue

        # ----------------------------------------------------
        # COUNTER
        # ----------------------------------------------------

        (
            ev,
            probabilities,
            known,
        ) = _qt_counter_ev(
            self,
            obs,
            candidate,
            turn,
        )

        # IMPORTANT:
        #
        # Do not allow a completely unseen opponent state to
        # generate a new counter using only global frequencies.
        #
        # Accept/reject actions remain available above.
        if not known:
            continue

        if (
            best_ev is None
            or ev > best_ev
        ):

            best_ev = ev
            best_action = candidate

    # --------------------------------------------------------
    # No usable candidate.
    # --------------------------------------------------------

    if (
        best_action is None
        or best_ev is None
    ):

        return None

    # --------------------------------------------------------
    # Conservative deployment gate.
    # --------------------------------------------------------

    if (
        best_ev
        < baseline_ev
        + 0.35
    ):

        return None

    # --------------------------------------------------------
    # Convert to engine action.
    # --------------------------------------------------------

    if (
        best_action[0]
        == "ACCEPT_BUY"
    ):

        return "ACCEPT_BUY"

    if (
        best_action[0]
        == "ACCEPT_SELL"
    ):

        return "ACCEPT_SELL"

    return (
        "COUNTER",
        int(
            best_action[1]
        ),
        int(
            best_action[2]
        ),
    )


# ============================================================
# END EMBEDDED RESEARCH LAYER
# ============================================================
'''


# ============================================================
# Find Bot.respond()
# ============================================================

def find_respond_method(
    source: str,
):

    """
    Find Bot.respond() without relying on comments or exact
    formatting.
    """

    class_match = re.search(
        r"(?m)^class\s+Bot\s*(?:\([^)]*\))?\s*:",
        source,
    )

    if class_match is None:

        raise RuntimeError(
            "Could not find `class Bot`."
        )

    class_start = (
        class_match.start()
    )

    respond_match = re.search(
        r"(?m)^    def\s+respond\s*\(",
        source[
            class_start:
        ],
    )

    if respond_match is None:

        raise RuntimeError(
            "Could not find `Bot.respond()`."
        )

    return (
        class_start
        + respond_match.start()
    )


# ============================================================
# Find final-turn condition
# ============================================================

def find_final_turn_if(
    source: str,
    respond_start: int,
):

    match = re.search(
        r"(?m)^        if\s+turn\s*==\s*self\.config\.N_TURNS\s*:",
        source[
            respond_start:
        ],
    )

    if match is None:

        raise RuntimeError(
            "Could not find the final-turn condition "
            "inside Bot.respond()."
        )

    return (
        respond_start
        + match.start()
    )


# ============================================================
# Inject runtime
# ============================================================

def inject_runtime(
    source: str,
    model_blob: str,
):

    """
    Inject:

        1. embedded research model/runtime before Bot
        2. research decision immediately before final-turn logic
    """

    # --------------------------------------------------------
    # Prevent accidental double patching.
    # --------------------------------------------------------

    if (
        "_QT_MODEL_DATA"
        in source
    ):

        raise RuntimeError(
            "Input bot already contains the research layer. "
            "Use the original adaptive_final(2).py."
        )

    # --------------------------------------------------------
    # Locate Bot.
    # --------------------------------------------------------

    class_match = re.search(
        r"(?m)^class\s+Bot\s*(?:\([^)]*\))?\s*:",
        source,
    )

    if class_match is None:

        raise RuntimeError(
            "Could not locate `class Bot`."
        )

    class_start = (
        class_match.start()
    )

    # --------------------------------------------------------
    # Build runtime.
    # --------------------------------------------------------

    runtime = (
        RUNTIME_CODE.replace(
            "__MODEL_DATA__",
            model_blob,
        )
    )

    # --------------------------------------------------------
    # Insert runtime before Bot.
    # --------------------------------------------------------

    source = (
        source[:class_start]

        + runtime

        + "\n\n\n"

        + source[class_start:]
    )

    # --------------------------------------------------------
    # Recalculate locations because source changed.
    # --------------------------------------------------------

    respond_start = (
        find_respond_method(
            source
        )
    )

    final_turn_pos = (
        find_final_turn_if(
            source,
            respond_start,
        )
    )

    # --------------------------------------------------------
    # Inject research layer before final-turn optimizer.
    # --------------------------------------------------------

    injection = """        # ----------------------------------------------------
        # PHASE 4C / PHASE 4D RESEARCH NEGOTIATION LAYER
        #
        # Runs only before the exact final-turn optimizer.
        #
        # If the research model does not produce a sufficiently
        # strong evidence-backed improvement, the original
        # Adaptive policy continues unchanged.
        # ----------------------------------------------------

        _research_action = _qt_research_response(
            self,
            obs,
            quote,
            turn,
        )

        if _research_action is not None:
            return _research_action

"""

    source = (
        source[:final_turn_pos]

        + injection

        + source[final_turn_pos:]
    )

    return source


# ============================================================
# Validate generated Python
# ============================================================

def validate_python(
    source: str,
):

    try:

        ast.parse(
            source
        )

    except SyntaxError as exc:

        raise RuntimeError(
            "Generated competition bot has a syntax error:\n"
            f"  line: {exc.lineno}\n"
            f"  offset: {exc.offset}\n"
            f"  message: {exc.msg}"
        ) from exc


# ============================================================
# Check forbidden imports / constructs
# ============================================================

def check_generated_source(
    source: str,
):

    """
    Local sanity check matching the important tournament
    restrictions discovered from the validator.
    """

    forbidden_imports = (
        "base64",
        "json",
        "zlib",
    )

    for name in forbidden_imports:

        pattern = (
            r"(?m)^\s*"
            r"(?:from\s+"
            + re.escape(name)
            + r"\s+import|import\s+"
            + re.escape(name)
            + r")"
        )

        if re.search(
            pattern,
            source,
        ):

            raise RuntimeError(
                "Generated bot still contains forbidden "
                f"import: {name}"
            )

    if "globals()" in source:

        raise RuntimeError(
            "Generated bot still contains globals()."
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build standalone AdaptiveFinal competition bot "
            "with embedded Phase 4C/4D opponent model."
        )
    )

    parser.add_argument(
        "--bot",
        type=Path,
        default=Path(
            "adaptive_final(2).py"
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "research/opponent/"
            "phase3_policy_dataset.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "adaptive_final_competition.py"
        ),
    )

    args = parser.parse_args()

    print(
        "=" * 72
    )

    print(
        "BUILDING STANDALONE "
        "COMPETITION BOT"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Bot:     {args.bot}"
    )

    print(
        f"Dataset: {args.dataset}"
    )

    print(
        f"Output:  {args.output}"
    )

    # --------------------------------------------------------
    # Verify files.
    # --------------------------------------------------------

    if not args.bot.exists():

        raise FileNotFoundError(
            f"Bot file not found: {args.bot}"
        )

    if not args.dataset.exists():

        raise FileNotFoundError(
            f"Dataset not found: {args.dataset}"
        )

    # --------------------------------------------------------
    # Load.
    # --------------------------------------------------------

    print()
    print(
        "Loading dataset..."
    )

    rows = load_rows(
        args.dataset
    )

    print(
        f"Total rows: "
        f"{len(rows):,}"
    )

    # --------------------------------------------------------
    # Fit.
    # --------------------------------------------------------

    print()
    print(
        "Fitting Phase-4C quote model..."
    )

    (
        tables,
        global_counts,
        response_rows,
    ) = fit_quote_model(
        rows
    )

    print(
        f"Response rows: "
        f"{response_rows:,}"
    )

    print(
        f"Learned states: "
        f"{len(tables):,}"
    )

    print(
        "Global counts:",
        dict(
            global_counts
        ),
    )

    # --------------------------------------------------------
    # Serialize.
    # --------------------------------------------------------

    print()
    print(
        "Serializing model as native Python..."
    )

    model_blob = serialize_model(
        tables,
        global_counts,
    )

    print(
        f"Embedded model source size: "
        f"{len(model_blob) / 1024:.1f} KB"
    )

    # --------------------------------------------------------
    # Read bot.
    # --------------------------------------------------------

    print()
    print(
        "Reading AdaptiveFinal..."
    )

    source = args.bot.read_text(
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Inject.
    # --------------------------------------------------------

    print(
        "Injecting research layer..."
    )

    output_source = inject_runtime(
        source,
        model_blob,
    )

    # --------------------------------------------------------
    # Validate syntax.
    # --------------------------------------------------------

    print(
        "Validating generated Python..."
    )

    validate_python(
        output_source
    )

    # --------------------------------------------------------
    # Validate tournament restrictions that caused the
    # previous failure.
    # --------------------------------------------------------

    print(
        "Checking tournament-safe constructs..."
    )

    check_generated_source(
        output_source
    )

    # --------------------------------------------------------
    # Write.
    # --------------------------------------------------------

    args.output.write_text(
        output_source,
        encoding="utf-8",
    )

    print()
    print(
        "=" * 72
    )

    print(
        "BUILD COMPLETE"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Output: "
        f"{args.output.resolve()}"
    )

    print()

    print(
        f"Embedded states: "
        f"{len(tables):,}"
    )

    print(
        f"Embedded model source: "
        f"{len(model_blob) / 1024:.1f} KB"
    )

    print()

    print(
        "Generated bot contains no:"
    )

    print(
        "  base64"
    )

    print(
        "  json"
    )

    print(
        "  zlib"
    )

    print(
        "  globals()"
    )

    print()

    print(
        "The 90 MB dataset is NOT required at runtime."
    )

    print()

    print(
        "Next:"
    )

    print(
        "  python -m py_compile "
        f'"{args.output}"'
    )

    print(
        "  python backtester.py --validate "
        f'"strategies/adaptive_competition.py"'
    )


if __name__ == "__main__":

    main()