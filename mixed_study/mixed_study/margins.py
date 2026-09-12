"""Section 3.2: margin-stability checks that test what the theorem says.

The bound is: if every candidate score moves by at most eps, the dense winner
stays the winner whenever its top-two margin exceeds 2*eps. The original check
used |gold margin| and tested "did the predicted answer change", which are
different statements once the dense answer is already wrong:

    dense [-10, 1, 0] -> pred 1;  compressed [-10, 0, 1] -> pred 2;  eps = 1
    |gold margin| = 11 > 2  but the argmax moved between two WRONG answers.

That satisfies the old violation test without contradicting any theorem. Two
correct statements are implemented here, on the same score and candidate set
the evaluation used.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence

from .records import Row


@dataclass
class MarginCheck:
    uid: str
    eps: float                      # max_k |S_c(k) - S_d(k)| over the candidates actually scored
    dense_top2_margin: float        # S_d(winner) - S_d(runner_up)
    argmax_certified: bool          # dense_top2_margin > 2*eps
    argmax_changed: bool
    argmax_violation: bool          # certified AND changed  -> would contradict the theorem
    gold_margin: float | None       # S_d(gold) - max_{k != gold} S_d(k); positive iff dense correct
    gold_certified: bool | None     # gold_margin > 2*eps
    gold_lost: bool | None          # dense correct, compressed not
    gold_violation: bool | None     # certified AND lost
    tie_at_top: bool                # dense top-two within 1e-9: theorem gives no guarantee


def _scores(r: Row, norm: str) -> Sequence[float]:
    return r.logprob_sum if norm == "sum" else r.logprob_mean


def check_pair(d: Row, c: Row, norm: str = "sum", tie_tol: float = 1e-9) -> MarginCheck:
    sd, sc = _scores(d, norm), _scores(c, norm)
    assert len(sd) == len(sc), f"{d.uid}: candidate count differs between dense and compressed"
    eps = max(abs(a - b) for a, b in zip(sd, sc))
    order = sorted(range(len(sd)), key=lambda i: -sd[i])
    win, run = order[0], (order[1] if len(order) > 1 else order[0])
    top2 = sd[win] - sd[run]
    tie = top2 <= tie_tol
    cwin = max(range(len(sc)), key=lambda i: sc[i])
    changed = cwin != win
    cert = (not tie) and top2 > 2 * eps
    gold_m = gold_cert = gold_lost = gold_viol = None
    if d.label is not None and 0 <= d.label < len(sd):
        others = [sd[i] for i in range(len(sd)) if i != d.label]
        gold_m = sd[d.label] - (max(others) if others else float("-inf"))
        gold_cert = gold_m > 2 * eps
        gold_lost = (win == d.label) and (cwin != d.label)
        gold_viol = bool(gold_cert and gold_lost)
    return MarginCheck(d.uid, eps, top2, cert, changed, bool(cert and changed),
                       gold_m, gold_cert, gold_lost, gold_viol, tie)


def check_all(pairs: Iterable[tuple[Row, Row]], norm: str = "sum") -> dict[str, Any]:
    checks = [check_pair(d, c, norm) for d, c in pairs]
    n = len(checks)
    cert = [x for x in checks if x.argmax_certified]
    gcert = [x for x in checks if x.gold_certified]
    out = {
        "n": n, "norm": norm,
        "argmax": {
            "n_certified": len(cert),
            "n_changed_overall": sum(x.argmax_changed for x in checks),
            "n_violations": sum(x.argmax_violation for x in checks),
            "changed_among_uncertified": sum(x.argmax_changed for x in checks if not x.argmax_certified),
        },
        "gold": {
            "n_with_label": sum(x.gold_margin is not None for x in checks),
            "n_dense_correct": sum(1 for x in checks if x.gold_margin is not None and x.gold_margin > 0),
            "n_certified": len(gcert),
            "n_lost_overall": sum(bool(x.gold_lost) for x in checks),
            "n_violations": sum(bool(x.gold_violation) for x in checks),
        },
        "ties_at_top": sum(x.tie_at_top for x in checks),
        "eps": {"mean": sum(x.eps for x in checks) / n if n else None,
                "max": max((x.eps for x in checks), default=None)},
        "violations": [asdict(x) for x in checks if x.argmax_violation or x.gold_violation][:50],
    }
    return out
