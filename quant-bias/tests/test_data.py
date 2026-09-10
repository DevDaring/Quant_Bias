import pytest
from quantbias.data import (split_for, check_split_disjoint, Example, counterfactual_pairs, pair_audit_checks,
                       _find_occupations, _parse_clusters, bbq_prompt, SPLIT_NAMES)


def test_split_deterministic_and_covering():
    a = split_for("bbq:Age:1", 7)
    assert a == split_for("bbq:Age:1", 7) and a in SPLIT_NAMES
    counts = {s: 0 for s in SPLIT_NAMES}
    for i in range(3000):
        counts[split_for(f"c{i}", 7)] += 1
    assert abs(counts["calibration"] / 3000 - 0.2) < 0.03
    assert abs(counts["final"] / 3000 - 0.5) < 0.03


def _ex(uid, cluster, split, group, label=0, bench="bbq", **meta):
    return Example(uid=uid, benchmark=bench, cluster_id=cluster, split=split, group=group,
                   prompt="p", candidates=[" a", " b", " c"], label=label, meta=meta)


def test_disjointness_check_detects_leak():
    ok = [_ex("1", "c1", "final", "g"), _ex("2", "c1", "final", "g"), _ex("3", "c2", "selection", "g")]
    assert check_split_disjoint(ok)["ok"]
    leak = ok + [_ex("4", "c1", "selection", "g")]
    r = check_split_disjoint(leak)
    assert not r["ok"] and "c1" in r["leaking"]


def test_winobias_parsing():
    toks = ["The", "janitor", "reprimanded", "the", "accountant", "because", "she", "made", "a", "mistake", "."]
    assert _parse_clusters(["3", "4", "6", "6"]) == [(3, 4), (6, 6)]
    occ = _find_occupations(toks)
    assert [o[2].lower() for o in occ] == ["the janitor", "the accountant"]


def test_counterfactual_pairs_and_audit():
    wb = [Example("wb:pro", "winobias", "wb:1", "final", "female", "p1", [" The janitor", " The accountant"], 1, meta={"condition": "pro"}),
          Example("wb:anti", "winobias", "wb:1", "final", "male", "p2", [" The janitor", " The accountant"], 1, meta={"condition": "anti"})]
    de = [Example("de:b", "discrim_eval", "de:0", "final", "male|white|60", "p", [" yes", " no"], None,
                  group_fields={"age": 60, "gender": "male", "race": "white"}, meta={"is_baseline": True}),
          Example("de:x", "discrim_eval", "de:0", "final", "female|Black|20", "p", [" yes", " no"], None,
                  group_fields={"age": 20, "gender": "female", "race": "Black"}, meta={"is_baseline": False})]
    pairs = counterfactual_pairs(wb + de)
    assert {p.changed for p in pairs} == {"pronoun", "age+gender+race"}
    chk = pair_audit_checks(pairs[0])
    assert chk["same_label"] and chk["same_candidates"] and chk["same_cluster"]


def test_bbq_prompt_formats():
    p, c = bbq_prompt("ctx", "q?", ["The man", "Unknown", "The woman"], "text")
    assert c == [" The man", " Unknown", " The woman"] and p.endswith("Answer:")
    p, c = bbq_prompt("ctx", "q?", ["x", "y", "z"], "letter")
    assert c == [" A", " B", " C"]


@pytest.mark.network
def test_real_loaders_small():
    from quantbias.data import load_bbq, load_winobias, load_discrim_eval
    bbq = load_bbq(1, categories="Sexual_orientation")
    assert bbq and all(e.meta["unknown_idx"] is not None for e in bbq)
    assert any(e.meta["target_idx"] is not None for e in bbq)
    wb = load_winobias(1, types=(1,), splits=("validation",))
    assert len(wb) > 700 and {e.meta["condition"] for e in wb} == {"pro", "anti"}
    de = load_discrim_eval(1, max_questions=2)
    assert de and any(e.meta["is_baseline"] for e in de)
