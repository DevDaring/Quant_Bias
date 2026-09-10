"""Bias-aware precision allocation under a byte budget (plan, Section 8).

The allocator only talks to the model through ``score_fn(PrecisionMap) ->
dict`` so that it can be unit-tested with a synthetic scorer, and through
``account_bytes`` for the constraint  C(b) <= M.

Objective (lower is better), frozen per run:
    J = w_H * H + w_A * A + w_gap * mean_abs_gap_change
subject to utility gates on accuracy loss and relative PPL increase against an
equally budgeted utility-only allocation.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Sequence

from .common import log
from .model_adapters import ModelAdapter
from .quantization import PrecisionMap, ByteFormat, account_bytes, DENSE_BITS


@dataclass
class Objective:
    w_H: float = 1.0
    w_A: float = 1.0
    w_gap: float = 1.0
    max_acc_loss: float = 0.01        # 1 percentage point vs utility-only baseline
    max_rel_ppl_increase: float = 0.05

    def value(self, m: dict[str, Any]) -> float:
        return (self.w_H * float(m.get("H", 0.0)) + self.w_A * float(m.get("A", 0.0))
                + self.w_gap * float(m.get("gap", 0.0)))

    def utility_ok(self, m: dict[str, Any], ref: dict[str, Any]) -> bool:
        acc_ok = (ref.get("acc") is None or m.get("acc") is None
                  or m["acc"] >= ref["acc"] - self.max_acc_loss)
        ppl_ok = (ref.get("ppl") is None or m.get("ppl") is None
                  or m["ppl"] <= ref["ppl"] * (1 + self.max_rel_ppl_increase))
        return bool(acc_ok and ppl_ok)


@dataclass
class SearchStep:
    step: int
    raised: str | None
    lowered: str | None
    bytes_total: int
    objective: float
    metrics: dict[str, Any]
    accepted: bool
    elapsed_s: float


@dataclass
class AllocationResult:
    name: str
    precision_map: dict[str, int]
    bytes_total: int
    budget: int
    objective: float
    metrics: dict[str, Any]
    steps: list[SearchStep] = field(default_factory=list)
    n_evaluations: int = 0
    search_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------
# Baseline allocations at matched bytes
# ----------------------------------------------------------------------------

def uniform_map(adapter: ModelAdapter, bits: int) -> PrecisionMap:
    return PrecisionMap.uniform(adapter, bits)


def budget_for_uniform(adapter: ModelAdapter, bits: int, fmt: ByteFormat | None = None) -> int:
    return account_bytes(adapter, uniform_map(adapter, bits), fmt).total


def ranked_map(adapter: ModelAdapter, scores: dict[str, float], budget: int, low: int = 4, high: int = 8,
               fmt: ByteFormat | None = None, descending: bool = True) -> PrecisionMap:
    """Start uniform ``low``; raise components to ``high`` in order of ``scores``
    while the byte budget allows. Used for ppl-only and margin-only baselines."""
    pm = PrecisionMap.uniform(adapter, low)
    order = sorted(scores, key=lambda k: scores[k], reverse=descending)
    for cid in order:
        pm[cid] = high
        if account_bytes(adapter, pm, fmt).total > budget:
            pm[cid] = low
    return pm


def random_map(adapter: ModelAdapter, budget: int, seed: int, low: int = 4, high: int = 8,
               fmt: ByteFormat | None = None) -> PrecisionMap:
    rng = random.Random(seed)
    scores = {c.id: rng.random() for c in adapter.components}
    return ranked_map(adapter, scores, budget, low, high, fmt)


# ----------------------------------------------------------------------------
# Greedy exchange
# ----------------------------------------------------------------------------

class GreedyAllocator:
    def __init__(self, adapter: ModelAdapter, score_fn: Callable[[PrecisionMap], dict[str, Any]],
                 budget: int, objective: Objective, shortlist: Sequence[str] | None = None,
                 lower_pool: Sequence[str] | None = None, bits_levels: Sequence[int] = (4, 8, 16),
                 fmt: ByteFormat | None = None, max_steps: int = 50, max_evals: int = 400,
                 utility_reference: dict[str, Any] | None = None, log_fn=log):
        """
        shortlist   components allowed to be *raised* (from propagation diagnostics)
        lower_pool  components allowed to be *lowered* to pay for a raise
                    (default: all components, ranked by utility regret if supplied)
        """
        self.adapter = adapter
        self.score_fn = score_fn
        self.budget = budget
        self.obj = objective
        self.shortlist = list(shortlist) if shortlist else [c.id for c in adapter.components]
        self.lower_pool = list(lower_pool) if lower_pool else [c.id for c in adapter.components]
        self.levels = sorted(bits_levels)
        self.fmt = fmt
        self.max_steps = max_steps
        self.max_evals = max_evals
        self.ref = utility_reference or {}
        self.log = log_fn or (lambda m: None)
        self.n_evals = 0
        self._cache: dict[str, dict[str, Any]] = {}

    def _score(self, pm: PrecisionMap) -> dict[str, Any]:
        h = pm.hash()
        if h not in self._cache:
            if self.n_evals >= self.max_evals:
                raise RuntimeError("evaluation budget exhausted")
            self._cache[h] = self.score_fn(pm)
            self.n_evals += 1
        return self._cache[h]

    def _bytes(self, pm: PrecisionMap) -> int:
        return account_bytes(self.adapter, pm, self.fmt).total

    def _next_up(self, b: int) -> int | None:
        higher = [l for l in self.levels if l > b]
        return higher[0] if higher else None

    def _next_down(self, b: int) -> int | None:
        lower = [l for l in self.levels if l < b]
        return lower[-1] if lower else None

    def run(self, initial: PrecisionMap, name: str = "greedy") -> AllocationResult:
        t0 = time.time()
        pm = PrecisionMap(dict(initial))
        if self._bytes(pm) > self.budget:
            raise ValueError("initial allocation exceeds budget")
        cur = self._score(pm)
        cur_J = self.obj.value(cur)
        steps: list[SearchStep] = []
        self.log(f"[{name}] start J={cur_J:.4f} bytes={self._bytes(pm)} budget={self.budget}")
        for step in range(self.max_steps):
            best = None  # (gain_per_byte, raised, lowered, new_pm, metrics, J)
            for cid in self.shortlist:
                up = self._next_up(pm.bits_of(cid))
                if up is None:
                    continue
                cand = PrecisionMap(dict(pm))
                cand[cid] = up
                added = self._bytes(cand) - self._bytes(pm)
                lowered = None
                if self._bytes(cand) > self.budget:
                    # pay for it: lower the cheapest-regret component in the pool
                    for lid in self.lower_pool:
                        if lid == cid:
                            continue
                        dn = self._next_down(cand.bits_of(lid))
                        if dn is None:
                            continue
                        trial = PrecisionMap(dict(cand))
                        trial[lid] = dn
                        if self._bytes(trial) <= self.budget:
                            cand, lowered = trial, lid
                            break
                    if lowered is None:
                        continue
                try:
                    m = self._score(cand)
                except RuntimeError:
                    break
                if not self.obj.utility_ok(m, self.ref):
                    continue
                J = self.obj.value(m)
                gain = cur_J - J
                per_byte = gain / max(1, added)
                if gain > 0 and (best is None or per_byte > best[0]):
                    best = (per_byte, cid, lowered, cand, m, J)
            elapsed = time.time() - t0
            if best is None:
                steps.append(SearchStep(step, None, None, self._bytes(pm), cur_J, cur, False, elapsed))
                self.log(f"[{name}] no improving feasible exchange at step {step}; stop")
                break
            _, cid, lowered, pm, cur, cur_J = best
            steps.append(SearchStep(step, cid, lowered, self._bytes(pm), cur_J, cur, True, elapsed))
            self.log(f"[{name}] step {step}: raise {cid}" + (f", lower {lowered}" if lowered else "")
                     + f" -> J={cur_J:.4f} bytes={self._bytes(pm)}")
        return AllocationResult(name=name, precision_map=dict(pm), bytes_total=self._bytes(pm), budget=self.budget,
                                objective=cur_J, metrics=cur, steps=steps, n_evaluations=self.n_evals,
                                search_seconds=time.time() - t0)


def evaluate_fixed_map(adapter: ModelAdapter, score_fn, pm: PrecisionMap, name: str, objective: Objective,
                       budget: int, fmt: ByteFormat | None = None) -> AllocationResult:
    m = score_fn(pm)
    return AllocationResult(name=name, precision_map=dict(pm), bytes_total=account_bytes(adapter, pm, fmt).total,
                            budget=budget, objective=objective.value(m), metrics=m, n_evaluations=1)
