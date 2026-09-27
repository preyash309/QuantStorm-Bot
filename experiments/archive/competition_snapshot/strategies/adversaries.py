from strategies.rational import Bot as RationalBot

class QuoteLiar(RationalBot):
    name = "QuoteLiar"
    def quote(self, obs):
        lo, hi = super().quote(obs)
        # Deliberately lie by 4 ticks to poison Bayesian anchors
        shift = 4 if self.rng.random() > 0.5 else -4
        
        # Keep within game bounds
        MAX = self.config.N_COINS
        lo = max(-MAX, min(lo + shift, MAX))
        hi = max(-MAX, min(hi + shift, MAX))
        if lo > hi: lo, hi = hi, lo
        
        return (lo, hi)

class TEHoarder(RationalBot):
    name = "TEHoarder"
    def bid(self, obs, offered):
        # Bid absolutely nothing R1-R3, then all-in on late-game FORESIGHT
        if obs.round >= 4 and "FORESIGHT" in offered:
            return {"FORESIGHT": obs.te_mine}
        return {}

class Turn6Forcer(RationalBot):
    name = "Turn6Forcer"
    def respond(self, obs, quote, turn):
        # Refuse to accept good prices; forcefully drive negotiation to the Turn 6 midpoint
        bid, ask = quote
        if turn < self.config.N_TURNS:
            w = max(self.config.final_cap(obs.round), (ask - bid) - self.config.MIN_REDUCTION)
            center = (bid + ask) // 2
            return ("COUNTER", center - w // 2, center + w - (w // 2))
        return super().respond(obs, quote, turn)