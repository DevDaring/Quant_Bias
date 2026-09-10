import math
import torch
from quantbias.data import Example
from quantbias.evaluate import score_candidates, bbq_metrics, winobias_metrics, discrim_metrics, harm_summary, ScoredExample


def _manual_logprob(adapter, prompt, cand):
    tok = adapter.tokenizer
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    j = tok(prompt + cand, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        lp = torch.log_softmax(adapter.model(torch.tensor([j])).logits[0].float(), -1)
    return sum(lp[i - 1, j[i]].item() for i in range(len(p), len(j)))


def test_score_matches_manual_and_permutation(tiny_adapter):
    a = tiny_adapter
    ex = Example("u", "bbq", "c", "final", "g", "Context: x.\nQuestion: who?\nAnswer:", [" The man", " Unknown", " The woman"], 2,
                 meta={"context_condition": "disambig", "unknown_idx": 1, "target_idx": 0, "question_polarity": "neg"})
    s = score_candidates(a, [ex], batch_size=2)[0]
    for i, c in enumerate(ex.candidates):
        assert math.isclose(s.logprob_sum[i], _manual_logprob(a, ex.prompt, c), abs_tol=1e-3)
    perm = Example("u2", "bbq", "c", "final", "g", ex.prompt, [ex.candidates[2], ex.candidates[0], ex.candidates[1]], 0, meta=ex.meta)
    s2 = score_candidates(a, [perm], batch_size=2)[0]
    assert math.isclose(s.logprob_sum[2], s2.logprob_sum[0], abs_tol=1e-4)
    assert not s.boundary_mismatch


def _se(uid, bench, group, pred, label, cluster="c", **meta):
    n = 3 if bench == "bbq" else 2
    return ScoredExample(uid, bench, cluster, "final", group, label, [0.0] * n, [0.0] * n, [1] * n, pred, 1.0,
                         (pred == label) if label is not None else None, False, {}, meta)


def test_bbq_official_scores():
    # disambig: 4 answered, 3 biased, 1 unknown -> s_DIS = 2*3/3 - 1 = 1.0 (all non-unknown are biased)
    rows = [_se(f"d{i}", "bbq", "F", pred=0, label=2, context_condition="disambig", unknown_idx=1, target_idx=0) for i in range(3)]
    rows.append(_se("d3", "bbq", "F", pred=1, label=2, context_condition="disambig", unknown_idx=1, target_idx=0))
    m = bbq_metrics(rows)
    assert m["disambig"]["bias_score"] == 1.0 and m["disambig"]["acc"] == 0.0
    assert m["disambig"]["stereotype_aligned_error_rate"] == 0.75
    amb = [_se(f"a{i}", "bbq", "F", pred=1, label=1, context_condition="ambig", unknown_idx=1, target_idx=0) for i in range(2)]
    amb.append(_se("a2", "bbq", "F", pred=0, label=1, context_condition="ambig", unknown_idx=1, target_idx=0))
    m = bbq_metrics(amb)
    assert math.isclose(m["ambig"]["acc"], 2 / 3) and math.isclose(m["ambig"]["bias_score"], (1 - 2 / 3) * 1.0)


def test_winobias_and_discrim_and_harm():
    wb = [_se("p1", "winobias", "female", 0, 0, condition="pro", type=1), _se("a1", "winobias", "male", 1, 0, condition="anti", type=1)]
    m = winobias_metrics(wb)
    assert m["acc_pro"] == 1.0 and m["acc_anti"] == 0.0 and m["gap_pro_minus_anti"] == 1.0
    de = [ScoredExample("b", "discrim_eval", "q", "final", "male|white|60", None, [0.0, -1.0], [0.0, -1.0], [1, 1], 0, 1.0, None, False,
                        {"age": 60, "gender": "male", "race": "white"}, {"decision_question_id": 0, "is_baseline": True}),
          ScoredExample("x", "discrim_eval", "q", "final", "female|Black|20", None, [-1.0, 0.0], [-1.0, 0.0], [1, 1], 1, 1.0, None, False,
                        {"age": 20, "gender": "female", "race": "Black"}, {"decision_question_id": 0, "is_baseline": False})]
    d = discrim_metrics(de)
    assert d["discrimination_score"]["gender"]["female"] < 0 < d["by_group"]["male|white|60"]["p_yes"]
    dense = [_se(f"g1{i}", "winobias", "g1", 0, 0) for i in range(30)] + [_se(f"g2{i}", "winobias", "g2", 0, 0) for i in range(30)]
    quant = [_se(f"g1{i}", "winobias", "g1", 0, 0) for i in range(30)] + [_se(f"g2{i}", "winobias", "g2", 1 if i < 6 else 0, 0) for i in range(30)]
    h = harm_summary(dense, quant, "winobias")
    assert math.isclose(h["delta_E"]["g2"], 0.2) and h["H_group"] == "g2" and math.isclose(h["A_disparity_increase"], 0.2)
