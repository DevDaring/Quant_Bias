"""§3.1-3.3: the corrected definitions must behave as the plan specifies."""
from mixed_study import harm, margins, pairs


def test_task_damage_excludes_unlabelled_and_splits_directions(rows):
    d = [rows("a", "bbq", "c1", "g", 0, [0, -1, -2], "disambig", 1, 2),     # dense correct
         rows("b", "bbq", "c1", "g", 0, [-1, 0, -2], "disambig", 1, 2),     # dense wrong
         rows("u", "discrim_eval", "q", "g", None, [0, -1])]                # unlabelled
    c = [rows("a", "bbq", "c1", "g", 0, [-1, 0, -2], "disambig", 1, 2),     # -> wrong (harmful)
         rows("b", "bbq", "c1", "g", 0, [0, -1, -2], "disambig", 1, 2),     # -> correct (beneficial)
         rows("u", "discrim_eval", "q", "g", None, [-1, 0])]
    t = harm.task_damage(zip(d, c))
    assert t.n_labelled == 2 and t.harmful == 1 and t.beneficial == 1
    r = t.rates()
    assert r["harmful_per_dense_correct"] == 1.0 and r["beneficial_per_dense_wrong"] == 1.0


def test_stereotype_damage_separates_target_aligned_from_other_errors(rows):
    # gold=0, target(stereotype)=1, unknown=2
    d = [rows("a", "bbq", "c", "g", 0, [0, -1, -2], "disambig", 1, 2),
         rows("b", "bbq", "c", "g", 0, [0, -1, -2], "disambig", 1, 2)]
    c = [rows("a", "bbq", "c", "g", 0, [-1, 0, -2], "disambig", 1, 2),     # -> target: stereotype error
         rows("b", "bbq", "c", "g", 0, [-1, -2, 0], "disambig", 1, 2)]     # -> unknown: other error
    s = harm.stereotype_damage(zip(d, c))
    assert s.new_stereotype_errors == 1 and s.new_other_errors == 1 and s.comp_unknown == 1


def test_group_disparity_reports_clusters_and_eligibility(rows):
    d, c = [], []
    for i in range(30):
        d.append(rows(f"g1_{i}", "bbq", f"cl{i % 5}", "G1", 0, [0, -1, -2], "disambig", 1, 2))
        c.append(rows(f"g1_{i}", "bbq", f"cl{i % 5}", "G1", 0, [0, -1, -2] if i % 3 else [-1, 0, -2], "disambig", 1, 2))
    for i in range(25):   # 25 rows but only ONE template -> restricted
        d.append(rows(f"g2_{i}", "bbq", "single", "G2", 0, [0, -1, -2], "disambig", 1, 2))
        c.append(rows(f"g2_{i}", "bbq", "single", "G2", 0, [0, -1, -2], "disambig", 1, 2))
    g = harm.group_disparity(zip(d, c), min_n=20, min_clusters=3)
    assert "G1" in g["eligible"] and g["eligible"]["G1"]["n_clusters"] == 5
    assert "G2" in g["restricted_descriptive_only"]
    assert g["H_worst_added_error"]["group"] == "G1"


def test_margin_counterexample_from_the_plan(rows):
    # dense [-10, 1, 0] -> pred 1 ; compressed [-10, 0, 1] -> pred 2 ; gold = 0 ; eps = 1
    d = rows("x", "bbq", "c", "g", 0, [-10, 1, 0], "disambig")
    c = rows("x", "bbq", "c", "g", 0, [-10, 0, 1], "disambig")
    m = margins.check_pair(d, c)
    assert m.eps == 1.0
    assert m.argmax_changed and not m.argmax_certified          # top-2 margin 1 <= 2*eps: no guarantee
    assert not m.argmax_violation                                 # so NOT a violation
    assert m.gold_margin == -11.0 and m.gold_certified is False   # gold was never winning
    assert not m.gold_violation


def test_margin_certified_and_stable_is_not_a_violation(rows):
    d = rows("y", "bbq", "c", "g", 0, [5, 0, -1], "disambig")
    c = rows("y", "bbq", "c", "g", 0, [4.5, 0.4, -1], "disambig")     # eps 0.5, margin 5 > 1
    m = margins.check_pair(d, c)
    assert m.argmax_certified and not m.argmax_changed and not m.argmax_violation
    assert m.gold_certified and not m.gold_lost


def test_pair_gap_uses_each_examples_own_label(rows):
    a = rows("a", "bbq", "c", "g", 0, [0, -3, -3], "disambig", ags=("X", "Y", "unk"))
    b = rows("b", "bbq", "c", "g", 1, [-3, 0, -3], "disambig", ags=("X", "Y", "unk"))
    # both are confident in THEIR OWN correct answer -> gap ~ 0 under the corrected estimand
    assert abs(pairs.correct_answer_gap(a, b)) < 1e-6
    st = pairs.classify(a, b)
    assert st.kind == "template_matched_unaudited" and not st.audited


def test_discrim_pairs_are_matched_by_construction(rows):
    base = rows("b", "discrim_eval", "q0", "m|w|60", None, [0, -1], is_baseline=True)
    other = rows("o", "discrim_eval", "q0", "f|b|20", None, [-1, 0], is_baseline=False)
    p = pairs.rebuild_pairs({"b": base, "o": other})
    assert len(p) == 1 and p[0][2].kind == "matched_by_construction"
    assert abs(pairs.shared_decision_gap(base, other) - (0.7310585786 - 0.2689414214)) < 1e-6
