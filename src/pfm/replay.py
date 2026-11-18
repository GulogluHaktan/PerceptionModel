"""Replay scheduler (plan step 9): which scenarios go into the next training round so that new conditions are learned without forgetting old ones.
A library entry is a scenario (its hash is the id) with the condition it covers, when it was added and the last golden-set score of the model on it.
Rules: (1) the real-data share never drops below `min_real` (decision 7: capped synthetic share, equal compute); (2) every condition seen before keeps at
least `min_per_condition` of the synthetic budget (replay); (3) conditions where the last score regressed get extra weight; (4) new scenarios get a
`new_share` of the synthetic budget; (5) a fixed regression suite (the first scenario of each condition, fixed seed) is returned separately and never trained on."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Entry:
    scenario_hash: str
    condition: str                 # e.g. "fog/fast/city"
    round_added: int
    last_score: float | None = None
    best_score: float | None = None
    frames: int = 60


@dataclass
class Library:
    entries: list[Entry] = field(default_factory=list)

    def add(self, e: Entry):
        if any(x.scenario_hash == e.scenario_hash for x in self.entries):
            raise ValueError(f"scenario {e.scenario_hash} already in the library")
        self.entries.append(e)

    def conditions(self):
        return sorted({e.condition for e in self.entries})

    def regression_suite(self):
        """First scenario of every condition: never used for training, used to detect forgetting."""
        seen, out = set(), []
        for e in self.entries:
            if e.condition not in seen:
                seen.add(e.condition); out.append(e.scenario_hash)
        return out


def plan_round(lib: Library, round_no: int, n_total: int, min_real: float = 0.3, new_share: float = 0.4, min_per_condition: float = 0.05):
    """Returns {"real": n, "synthetic": {scenario_hash: n_frames}, "suite": [hashes], "n_synthetic": n}. Frame counts sum to n_total."""
    if not 0 <= min_real < 1 or not 0 < new_share <= 1:
        raise ValueError("bad shares")
    suite = set(lib.regression_suite())
    pool = [e for e in lib.entries if e.scenario_hash not in suite]
    n_synth = int(n_total * (1 - min_real)) if pool else 0
    new = [e for e in pool if e.round_added == round_no]; old = [e for e in pool if e.round_added < round_no]
    budget = {}
    new_budget = int(n_synth * (new_share if old else 1.0)) if new else 0
    for e in new:
        budget[e.scenario_hash] = new_budget // len(new)
    rest = n_synth - sum(budget.values())
    conds = sorted({e.condition for e in old})
    floor = int(n_synth * min_per_condition)
    weights = {}
    for e in old:
        drop = 0.0 if e.last_score is None or e.best_score is None else max(0.0, e.best_score - e.last_score)
        weights[e.scenario_hash] = 1.0 + 10.0 * drop                   # a regressed condition is replayed more
    for c in conds:
        group = [e for e in old if e.condition == c]
        for e in group:
            budget[e.scenario_hash] = budget.get(e.scenario_hash, 0) + floor // len(group)
    rest = n_synth - sum(budget.values())
    if old and rest > 0:
        tot = sum(weights.values())
        for e in old:
            budget[e.scenario_hash] = budget.get(e.scenario_hash, 0) + int(rest * weights[e.scenario_hash] / tot)
    n_syn = sum(budget.values())
    return {"real": n_total - n_syn, "synthetic": budget, "suite": sorted(suite), "n_synthetic": n_syn}
