from __future__ import annotations

class ResearchPowerState:
    def __init__(self):
        self.current_round = 0
        self.round_powers = [set(), set()]
        self.persistent = [set(), set()]
        self.used = set()
        self.substitute_rounds = [set(), set()]

    def new_round(self, r):
        self.current_round = int(r)
        self.round_powers[0].clear()
        self.round_powers[1].clear()

    def acquire(self, seat, name, r, config):
        spec = config.POWERS[name]

        if spec.get("once_per_deal", False):
            if name in self.used:
                return
            self.used.add(name)
            if name == "STEALTH_ROCK":
                self.persistent[seat].add(name)

        self.round_powers[seat].add(name)

        if name == "SUBSTITUTE":
            self.substitute_rounds[seat].add(int(r))

    def consumed(self, name, config):
        return (
            config.POWERS[name].get("once_per_deal", False)
            and name in self.used
        )

    def active(self, seat, r):
        current = self.round_powers[seat] if int(r) == self.current_round else set()
        return frozenset(current | self.persistent[seat])
