# Name: Research example
# College: Example institution
# Roll Number: REPLACE_WITH_YOUR_ROLL_NUMBER

"""
AdaptiveFinal — standalone QuantStorm submission.

Optimized to prevent adverse selection, avoid early-round Maker sniping,
conserve TE for late-game FORESIGHT, and discount poisoned opponent quotes.
Incorporates Dynamic Shading, Mixed-Strategy Obfuscation, and Power-Aware Stalling.
"""

import random
import math

# ============================================================
# Calibrated & Tuned Strategy Parameters
# ============================================================

POWER_VALUES = {
    "FORESIGHT":    {1: 0.76, 2: 1.16, 3: 1.48, 4: 1.97, 5: 2.02},
    "TRICK_ROOM":   {1: 1.14, 2: 0.00, 3: 0.00, 4: 0.60, 5: 0.52},
    "SUBSTITUTE":   {1: 1.46, 2: 1.15, 3: 0.95, 4: 0.57, 5: 0.29},
    "STEALTH_ROCK": {1: 1.51, 2: 0.75, 3: 0.75, 4: 0.75, 5: 0.00},
    "TRANSFORM":    {1: 1.58, 2: 1.24, 3: 1.31, 4: 0.00, 5: 0.00},
}

SHADE = 0.60
DENIAL_WEIGHT = 0.40  
FLAT_THRESHOLD = 1
OPP_FLAT_THRESHOLD = 2.0

class Bot:
    name = "AdaptiveFinal"

    def reset(self, seat, config, seed):
        self.seat = seat
        self.config = config
        self.rng = random.Random(seed)
        self._anchor = {}
        self._opp_anchor = {}

    def _get_anchor(self, obs, quote):
        r = obs.round
        if r not in self._anchor:
            if not obs.is_maker and quote is not None:
                # Information Extraction: Apply Bayesian discount to opponent midpoint
                raw_midpoint = (quote[0] + quote[1]) / 2.0
                
                # Trust late-round quotes heavily, but discount early quotes
                alpha = {1: 0.50, 2: 0.65, 3: 0.80, 4: 0.90, 5: 0.95}.get(r, 0.50)
                discounted = raw_midpoint * alpha
                
                # Clamp implied coins to mathematically possible bounds
                max_revealed = 4 * r
                clamped = max(-max_revealed, min(max_revealed, discounted))
                
                self._anchor[r] = clamped
                self._opp_anchor[r] = clamped
            else:
                self._anchor[r] = 0.0
        return self._anchor[r]

    def _opponent_k(self, obs):
        earlier = [k for k in self._opp_anchor if k < obs.round]
        if not earlier:
            return None
        return self._opp_anchor[max(earlier)]

    def _value(self, obs, quote=None):
        if obs.is_maker or quote is None:
            return float(obs.k_mine + sum(obs.foresight))
        
        anchor = self._get_anchor(obs, quote)
        
        if obs.foresight:
            # Optimal blend of exact leaked coins and inferred remaining coins
            n_seen = len(obs.foresight)
            n_total = 4 * obs.round
            weight_exact = n_seen / n_total if n_total > 0 else 0
            weight_inferred = 1.0 - weight_exact
            anchor = (weight_exact * sum(obs.foresight)) + (weight_inferred * anchor)
            
        return float(anchor + obs.k_mine)

    def _power_value(self, obs, name):
        return POWER_VALUES.get(name, {}).get(obs.round, 0.5)

    def _transform_value(self, obs):
        swap = self._power_value(obs, "TRANSFORM")
        if abs(obs.k_mine) <= FLAT_THRESHOLD:
            return swap
        opp_k = self._opponent_k(obs)
        if opp_k is not None and abs(opp_k) <= OPP_FLAT_THRESHOLD:
            return swap * DENIAL_WEIGHT
        return 0.0

    def bid(self, obs, offered):
        if not offered or obs.te_mine <= 0:
            return {}
            
        # Auction Theory: Dynamic Shading based on TE parity
        te_advantage = obs.te_mine - obs.te_theirs
        dynamic_shade = min(0.85, max(0.35, SHADE + (te_advantage * 0.015)))

        out = {}
        for name in offered:
            v = self._transform_value(obs) if name == "TRANSFORM" else self._power_value(obs, name)
            if v <= 0: continue
            
            fair_te = v / self.config.TE_SALVAGE
            calculated_bid = int(fair_te * dynamic_shade)
            
            # Mechanism Design: Exploit TE monopolies
            if calculated_bid > obs.te_theirs and obs.te_mine > obs.te_theirs:
                calculated_bid = obs.te_theirs + 1
                
            bid_amount = max(0, min(calculated_bid, obs.te_mine))
            out[name] = bid_amount

        # TE Preservation Pacing 
        if obs.round <= 3 and obs.te_mine > 6:
            for name in out:
                out[name] = min(out[name], obs.te_mine - 6)

        return out

    def quote(self, obs):
        v = round(self._value(obs))
        
        # Mixed Strategy: Inject noise into the center to break opponent mapping
        if obs.round > 1:
            skew = self.rng.choice([-1, 0, 0, 1])
            v += skew
        
        # Maker Sniping Protection
        if obs.round <= 2:
            width = min(obs.spread_cap, obs.final_cap + 2)
        else:
            width = obs.final_cap
            
        lo = v - width // 2
        return (lo, lo + width)

    def respond(self, obs, quote, turn):
        bid, ask = quote
        v = self._value(obs, quote)
        
        # ============================================================
        # EXACT TURN 6 EVALUATION
        # ============================================================
        if turn == self.config.N_TURNS:
            my_shift = sum(int(self.config.POWERS[pw]["magnitude"]) for pw in ("TRICK_ROOM", "STEALTH_ROCK") if pw in obs.powers_mine and pw in self.config.POWERS)
            opp_shift = sum(int(self.config.POWERS[pw]["magnitude"]) for pw in ("TRICK_ROOM", "STEALTH_ROCK") if pw in obs.powers_theirs and pw in self.config.POWERS)
            net_shift = my_shift - opp_shift
            
            old_width = ask - bid
            max_width = min(old_width, max(self.config.final_cap(obs.round), old_width - self.config.MIN_REDUCTION))
            new_ask = ask
            new_bid = new_ask - max_width
            forced_price = ((new_bid + new_ask) // 2) + net_shift
            
            buy_ev = v - ask
            sell_ev = bid - v
            force_ev = forced_price - v - self.config.FORCED_FILL_FEE
            
            best = max(buy_ev, sell_ev, force_ev)
            if best == buy_ev: return "ACCEPT_BUY"
            if best == sell_ev: return "ACCEPT_SELL"
            return ("COUNTER", new_bid, new_ask)

        # ============================================================
        # TURNS 2 TO 5: DYNAMIC MARGIN OF SAFETY & STALLING
        # ============================================================
        edge_buy = v - ask
        edge_sell = bid - v
        
        unknown = self.config.unknown_to_both(obs.round)
        edge_margin = 0.25 * math.sqrt(unknown) if unknown > 0 else 0.0
        
        if "SUBSTITUTE" in obs.powers_mine:
            edge_margin = max(0.0, edge_margin - 1.0)
            
        # Power-Aware Stalling: Demand heavy edge if holding a midpoint advantage
        my_shift = sum(int(self.config.POWERS[pw]["magnitude"]) for pw in ("TRICK_ROOM", "STEALTH_ROCK") if pw in obs.powers_mine and pw in self.config.POWERS)
        if my_shift > 0:
            edge_margin += (my_shift * 0.75)
            
        if edge_buy >= edge_margin and edge_buy >= edge_sell:
            return "ACCEPT_BUY"
        if edge_sell >= edge_margin:
            return "ACCEPT_SELL"

        # Safe Counter strategy with Mixed-Strategy Blurring
        w = max(0, (ask - bid) - self.config.MIN_REDUCTION)
        blur = self.rng.choice([-1, 0, 0, 1])
        center = max(bid, min(round(v) + blur, ask - w))
        
        return ("COUNTER", center, center + w)

    def use_transform(self, obs):
        return abs(obs.k_mine) <= FLAT_THRESHOLD