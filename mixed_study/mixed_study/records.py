"""Load the per-example records E1/E7 wrote, paired dense-vs-compressed.

A record row is one scored example under one configuration. Pairing is by
``uid`` within a model, dense against every compressed config. Nothing here
re-scores anything; it only reads what the GPU run stored.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .common import QB_RESULTS, read_jsonl


@dataclass(frozen=True)
class Row:
    uid: str
    benchmark: str
    cluster: str
    group: str
    label: int | None
    pred: int
    correct: bool | None
    logprob_sum: tuple[float, ...]
    logprob_mean: tuple[float, ...]
    margin: float
    context_condition: str | None
    target_idx: int | None
    unknown_idx: int | None
    category: str | None
    answer_groups: tuple[str, ...]
    group_fields: dict
    meta: dict

    @staticmethod
    def from_record(r: dict) -> "Row":
        m = r.get("meta") or {}
        gf = r.get("group_fields") or {}
        return Row(
            uid=r["uid"], benchmark=r["benchmark"], cluster=r["cluster_id"], group=r["group"],
            label=r.get("label"), pred=int(r["pred"]), correct=r.get("correct"),
            logprob_sum=tuple(r["logprob_sum"]), logprob_mean=tuple(r["logprob_mean"]),
            margin=float(r.get("margin", 0.0)),
            context_condition=m.get("context_condition"), target_idx=m.get("target_idx"),
            unknown_idx=m.get("unknown_idx"), category=gf.get("category"),
            answer_groups=tuple(gf.get("answer_groups") or ()), group_fields=gf, meta=m)

    def pred_under(self, norm: str) -> int:
        """Prediction under the declared scoring rule (sum or mean log-prob)."""
        s = self.logprob_sum if norm == "sum" else self.logprob_mean
        return max(range(len(s)), key=lambda i: s[i])

    def correct_under(self, norm: str) -> bool | None:
        if self.label is None:
            return None
        return self.pred_under(norm) == self.label


def load_config_rows(exp: str, tag: str, config: str) -> dict[str, Row]:
    """{uid: Row} for one model/config, e.g. ('e1','mistral_7b_v0_1','gptq4')."""
    f = QB_RESULTS / exp / tag / f"records_{config}.jsonl"
    if not f.exists():
        raise FileNotFoundError(f)
    return {r["uid"]: Row.from_record(r) for r in read_jsonl(f)}


def available_configs(exp: str, tag: str) -> list[str]:
    d = QB_RESULTS / exp / tag
    return sorted(p.stem[len("records_"):] for p in d.glob("records_*.jsonl"))


@dataclass
class Paired:
    """Dense and compressed rows for the same examples."""
    tag: str
    config: str
    dense: dict[str, Row]
    comp: dict[str, Row]

    @property
    def uids(self) -> list[str]:
        return sorted(set(self.dense) & set(self.comp))

    def rows(self, benchmark: str | None = None) -> Iterable[tuple[Row, Row]]:
        for u in self.uids:
            d = self.dense[u]
            if benchmark is None or d.benchmark == benchmark:
                yield d, self.comp[u]


def load_paired(exp: str, tag: str, config: str, dense_config: str = "dense") -> Paired:
    return Paired(tag, config, load_config_rows(exp, tag, dense_config), load_config_rows(exp, tag, config))
