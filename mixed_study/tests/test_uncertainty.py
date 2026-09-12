import numpy as np
from mixed_study import uncertainty as U


def test_multiplier_bootstrap_keeps_every_draw_and_every_group():
    rng = np.random.default_rng(0)
    n = 120
    clusters = [f"c{i // 4}" for i in range(n)]
    group = np.array(["A"] * 100 + ["B"] * 20)          # B is small: the source bootstrap would drop it
    dense = rng.random(n) < 0.8
    comp = dense & ~((group == "B") & (rng.random(n) < 0.4))
    r = U.multiplier_bootstrap(U.weighted_group_error_delta(dense, comp, group, "B"), clusters, n_boot=300)
    assert r["n_boot_valid"] == 300                    # nothing discarded
    assert r["ci_low"] > 0                             # detects the planted harm on the small group


def test_paired_difference_beats_overlapping_intervals():
    rng = np.random.default_rng(1)
    n = 400
    clusters = [f"c{i // 5}" for i in range(n)]
    a = rng.random(n) < 0.80
    b = a.copy(); b[rng.random(n) < 0.06] = False       # B slightly, consistently worse
    r = U.paired_method_difference(a, b, clusters, n_boot=400)
    assert r["ci_low"] > 0 and r["interpretation"] == "A better than B"


def test_coverage_simulation_runs_and_reports_verdict():
    clusters = [f"c{i // 6}" for i in range(180)]
    group = ["A" if i % 3 else "B" for i in range(180)]
    c = U.coverage_simulation(clusters, group, n_sim=20, n_boot=60)
    assert 0 <= c["empirical_coverage"] <= 1 and c["verdict"]
