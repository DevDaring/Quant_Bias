"""Focused tests required before GPU execution (plan section 9.1): data protocol,
outcome definitions on hand-checked rows, bootstrap support, packed-record
alignment, resumability. No model is loaded here."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from next_study import synthbias as S
from next_study import analysis as A
from next_study.common import DATA, write_json, read_json, seed_from


def _rows():
    p = DATA / "synthbias_canonical.jsonl"
    if not p.exists():
        pytest.skip("run `python -m next_study.run ingest` first")
    return S.load_rows(p)


def test_row_ids_deterministic_and_unique():
    a = S.row_uid("The nurse met the doctor because she was late.", "nurse", "doctor", "she", "type2", "pro")
    b = S.row_uid("The nurse met the doctor because she was late.", "nurse", "doctor", "she", "type2", "pro")
    c = S.row_uid("The nurse met the doctor because he was late.", "nurse", "doctor", "he", "type2", "anti")
    assert a == b and a != c and a.startswith("sb-")
    rows = _rows()
    assert len({r.uid for r in rows}) == len(rows)


def test_split_assignment_deterministic_and_disjoint():
    rows = _rows()
    m1 = S.assign_splits(rows, seed=20260914)
    m2 = S.assign_splits(rows, seed=20260914)
    assert m1["assignment"] == m2["assignment"]
    assert m1["disjoint"] and all(v == 0 for v in m1["cluster_overlap"].values())
    assert set(m1["assignment"].values()) == {"selection", "mechanism", "final"}
    fr = {k: v["clusters"] for k, v in m1["counts"].items()}
    total = sum(fr.values())
    assert abs(fr["final"] / total - 0.70) < 0.03 and abs(fr["selection"] / total - 0.10) < 0.03


def test_counterparts_and_occupation_pairs_stay_together():
    rows = _rows()
    by = {r.uid: r for r in rows}
    linked = [r for r in rows if r.counterpart_uid]
    assert len(linked) > 0.9 * len(rows)
    assert all(by[r.counterpart_uid].split == r.split for r in linked)
    split_of_cluster = {}
    for r in rows:
        assert split_of_cluster.setdefault(r.cluster, r.split) == r.split


def test_candidate_order_counterbalanced():
    rows = _rows()
    t2 = S.type2(rows)
    share = np.mean([S.candidate_order(r.uid)[0] for r in t2])
    assert 0.47 < share < 0.53
    r = t2[0]
    text, cands, gold = S.render("P2", r)
    assert cands[gold] == r.occ_1 and set(cands) == {r.occ_1, r.occ_2}
    assert f"Choices: {cands[0]}; {cands[1]}" in text
    text_s, cands_s, gold_s = S.render("P2", r, swap=True)
    assert cands_s == cands[::-1] and gold_s == 1 - gold and f"Choices: {cands_s[0]}; {cands_s[1]}" in text_s


def test_damage_definitions_on_hand_checked_rows():
    # four dense-correct rows (2 anti), one dense-wrong anti row
    dense = [
        {"uid": "a", "cluster": "c1", "stereo": "anti", "correct": True, "gold_position": 0},
        {"uid": "b", "cluster": "c1", "stereo": "anti", "correct": True, "gold_position": 1},
        {"uid": "c", "cluster": "c2", "stereo": "pro", "correct": True, "gold_position": 0},
        {"uid": "d", "cluster": "c2", "stereo": "pro", "correct": True, "gold_position": 1},
        {"uid": "e", "cluster": "c3", "stereo": "anti", "correct": False, "gold_position": 0},
    ]
    comp = [
        {"uid": "a", "correct": False, "pred_other": True},    # anti, lost to the stereotype -> D_stereo event
        {"uid": "b", "correct": True, "pred_other": False},
        {"uid": "c", "correct": False, "pred_other": True},    # pro, lost -> task damage only
        {"uid": "d", "correct": True, "pred_other": False},
        {"uid": "e", "correct": True, "pred_other": False},    # beneficial transition, not in C
    ]
    fns, counts = A._outcome_fns(dense, comp)
    w = np.ones(5)
    assert fns["D_task"](w) == pytest.approx(2 / 4)
    assert fns["D_stereo"](w) == pytest.approx(1 / 2)
    # Delta_G: pro acc dense 1.0 -> 0.5 ; anti acc dense 2/3 -> 2/3 ; gap change = (0.5-2/3) - (1-2/3) = -0.5
    assert fns["Delta_G"](w) == pytest.approx(-0.5)
    assert counts == {"n_dense_correct": 4, "n_anti_dense_correct": 2, "task_damage_events": 2, "stereo_events": 1,
                      "n_pro": 2, "n_anti": 3, "n_pos0": 3, "n_pos1": 2}


def test_bootstrap_preserves_support_and_pairs_draws():
    clusters = ["c1"] * 10 + ["c2"] * 10 + ["c3"] * 2
    boot = A.ClusterBoot(clusters, n_boot=300, seed=1)
    for b in range(300):
        w = boot.weights(b)
        assert (w > 0).all() and abs(w[:10].mean() - w[0]) < 1e-12   # every cluster present, cluster-constant weights
    # same draws for two statistics -> paired difference of identical stats is exactly zero
    x = np.random.default_rng(0).normal(size=22)
    d1 = boot.draws(lambda w: float(np.sum(w * x) / np.sum(w)))
    d2 = boot.draws(lambda w: float(np.sum(w * x) / np.sum(w)))
    assert np.all(d1 == d2)
    s = A.summarise(float(x.mean()), d1, one_sided_null="<=0")
    assert s["n_boot_valid"] == 300 and s["ci_low"] <= s["estimate"] <= s["ci_high"]


def test_holm_correction():
    out = A.holm({"a": 0.01, "b": 0.04, "c": 0.5, "d": None})
    assert out["a"]["p_holm"] == pytest.approx(0.04) and out["a"]["reject_0.05"]
    assert out["b"]["p_holm"] == pytest.approx(0.12) and not out["b"]["reject_0.05"]
    assert out["d"]["undefined"] and not out["d"]["reject_0.05"]


def test_packed_records_align_by_row_id():
    dense = [{"uid": u, "cluster": "c", "stereo": "anti", "correct": True, "gold_position": 0} for u in "abc"]
    packed = [{"uid": u, "correct": True, "pred_other": False} for u in "cab"]
    d, q = A.align(dense, packed)
    assert [r["uid"] for r in d] == [r["uid"] for r in q] == ["a", "b", "c"]
    with pytest.raises(AssertionError):
        A.align(dense, packed[:2])


def test_counter_based_seed_is_stable():
    assert seed_from("rev", "sb-1", 3, "rtn4", 0) == seed_from("rev", "sb-1", 3, "rtn4", 0)
    assert seed_from("rev", "sb-1", 3, "rtn4", 0) != seed_from("rev", "sb-1", 3, "rtn4", 1)


def test_atomic_write_and_resume_marker(tmp_path):
    p = tmp_path / "x" / "y.json"
    write_json(p, {"a": 1})
    assert read_json(p) == {"a": 1} and not p.with_suffix(".json.tmp").exists()
    write_json(p, {"a": 2})
    assert read_json(p) == {"a": 2}                     # replaced, never appended
