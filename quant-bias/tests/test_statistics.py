import numpy as np
from quantbias.statistics import paired_cluster_bootstrap, holm, tost_equivalence, group_ci_table


def test_bootstrap_recovers_effect():
    rng = np.random.default_rng(0)
    n = 400
    clusters = [f"c{i // 4}" for i in range(n)]
    cd = rng.random(n) < 0.8
    cq = cd.copy()
    flip = rng.random(n) < 0.15
    cq[flip & cd] = False                                      # ~12 pp more error
    d = (1 - cq.astype(float)) - (1 - cd.astype(float))
    r = paired_cluster_bootstrap(lambda rows: float(d[rows].mean()), clusters, n_boot=500, seed=1)
    assert r["ci_low"] > 0 and r["p_value"] < 0.05 and abs(r["estimate"] - d.mean()) < 1e-9


def test_holm_and_tost():
    h = holm({"a": 0.001, "b": 0.02, "c": 0.5})
    assert h["a"]["reject"] and not h["c"]["reject"] and h["b"]["p_holm"] >= 0.04
    diffs = np.random.default_rng(0).normal(0, 0.01, 300)
    t = tost_equivalence(diffs, [f"c{i // 3}" for i in range(300)], margin=0.01, n_boot=300)
    assert t["equivalent"]


def test_group_ci_table_structure():
    rng = np.random.default_rng(2)
    n = 240
    g = np.array(["A", "B", "C"] * (n // 3))
    cl = [f"c{i // 6}" for i in range(n)]
    cd = rng.random(n) < 0.7
    cq = cd & ~((g == "B") & (rng.random(n) < 0.3))
    t = group_ci_table(cd, cq, g, cl, min_n=20, n_boot=300)
    assert set(t["groups"]) == {"A", "B", "C"} and t["H"]["estimate"] > 0 and "mde_delta_E" in t
    assert t["groups"]["B"]["estimate"] > t["groups"]["A"]["estimate"]
