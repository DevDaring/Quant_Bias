"""B1 and directional prediction, live-fired on a tiny random model."""
import torch
from mixed_study import residuals as R, directional as D
from quantbias.quantization import Quantizer, PrecisionMap


def _batch(a, texts):
    tok = a.tokenizer
    enc = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    L = max(map(len, enc))
    ids = torch.full((len(enc), L), tok.pad_token_id); mask = torch.zeros((len(enc), L), dtype=torch.long)
    for i, e in enumerate(enc):
        ids[i, :len(e)] = torch.tensor(e); mask[i, :len(e)] = 1
    return {"input_ids": ids, "attention_mask": mask}, mask


def test_boundary_index_is_explicit(tiny):
    b = R.Boundary(-1); assert b.state_index == 0          # embeddings
    b = R.Boundary(2);  assert b.state_index == 3          # block 2 output
    st = R.capture_states(tiny, _batch(tiny, ["a b c"])[0])
    assert len(st) == tiny.n_layers + 1


def test_injection_changes_only_downstream(tiny):
    batch, mask = _batch(tiny, ["the nurse said", "a doctor spoke"])
    dense = R.capture_states(tiny, batch)
    b = R.Boundary(1)
    delta = torch.randn_like(dense[b.state_index]) * 0.1
    with R.inject(tiny, b, delta):
        pert = R.capture_states(tiny, batch)
    for j in range(len(dense)):
        same = torch.allclose(dense[j], pert[j], atol=1e-6)
        assert same == (j < b.state_index), f"state {j}: same={same}"
    assert torch.allclose(pert[b.state_index] - dense[b.state_index], delta, atol=1e-5)


def test_norm_matched_random_matches_per_token_norm(tiny):
    batch, mask = _batch(tiny, ["one two three", "four"])
    dense = R.capture_states(tiny, batch)
    res = torch.randn_like(dense[2]) * 0.3
    rnd = R.norm_matched_random(res, seed=3, mask=mask)
    n1, n2 = R.per_token_norm(res, mask), R.per_token_norm(rnd, mask)
    assert torch.allclose(n1, n2, atol=1e-5)
    # the mask must be moved to the residual's device inside the function, not assumed to match
    assert torch.allclose(R.norm_matched_random(res, seed=3, mask=mask.to(torch.int64)), rnd)
    assert not torch.allclose(res, rnd)


def test_run_sources_reports_actual_reversed_random(tiny):
    q = Quantizer(tiny)
    batch, mask = _batch(tiny, ["the cat sat on the mat", "a dog ran"])
    b = R.Boundary(1)
    comp_ids = [c.id for c in tiny.components_of(1)]
    dense, res = R.residual_at(tiny, q, PrecisionMap.single_site(tiny, comp_ids, 3), batch, b)
    assert res.abs().sum() > 0
    srcs = R.make_sources(res, n_random=2, mask=mask)
    props = R.run_sources(tiny, b, srcs, batch, mask)
    assert set(props) == {"actual", "sign_reversed", "random_0", "random_1"}
    a = props["actual"]
    assert a.cos_final_vs_actual is None and props["random_0"].cos_final_vs_actual is not None
    assert abs(props["sign_reversed"].local_abs - a.local_abs) < 1e-4       # same magnitude
    assert len(a.per_layer_rel) == len(dense_states := R.capture_states(tiny, batch)) - b.state_index
    assert all(q.is_restored(c.id) for c in tiny.components)


def test_directional_first_order_matches_finite_difference_for_small_residual(tiny):
    from quantbias.data import Example
    ex = Example("u", "bbq", "c", "final", "g", "The answer is", [" yes", " no", " maybe"], 0)
    b = R.Boundary(1)
    cg = D.contrast_and_gradient(tiny, ex, b)
    assert cg["grad_contrast"].shape[0] == len(cg["prompt_ids"])
    # a tiny residual along the gradient: first-order estimate should be close to the actual change
    g = cg["grad_contrast"]
    rp = 1e-3 * g / (g.norm() + 1e-12)
    out = D.predict_and_verify(tiny, cg, b, rp)
    assert out["predicted_delta_s"] > 0
    assert abs(out["predicted_delta_s"] - out["actual_delta_s"]) < 0.25 * abs(out["predicted_delta_s"]) + 1e-4


def test_restoration_arms_are_equal_cost(tiny):
    from mixed_study.matched_restoration import arms
    from quantbias.quantization import account_bytes
    ids = [c.id for c in tiny.components]
    A = arms(tiny, predicted=ids[:2], utility_ranked=list(reversed(ids)), k=2, n_random=3)
    pb = A["predicted"]["bytes"]
    for n, a in A.items():
        if n == "uniform_start":
            continue
        assert a["n_restored"] == 2
        assert a["buckets"] == A["predicted"]["buckets"]       # same kind & size class
        assert abs(a["bytes"] - pb) <= 0.01 * pb               # equal cost within 1%


def test_propagation_boundary_drift_equals_injected_delta(tiny):
    """The relative drift at the injection boundary itself must equal ||delta||/||h||,
    which the old output_hidden_states path silently reported as zero."""
    from mixed_study.residuals import _forward_capture
    batch, mask = _batch(tiny, ["the nurse said", "a doctor spoke"])
    dense, dl = _forward_capture(tiny, batch)
    b = R.Boundary(1)
    delta = torch.randn_like(dense[b.state_index]) * 0.05
    last = (mask.sum(1) - 1).long(); rows = torch.arange(mask.shape[0])
    p, _ = R.propagate(tiny, b, delta, batch, dense, dl[rows, last], mask)
    m = mask.float(); ntok = m.sum()
    expect = float(((delta.norm(dim=-1) / (dense[b.state_index].norm(dim=-1) + 1e-8)) * m).sum() / ntok)
    assert abs(p.per_layer_rel[0] - expect) < 1e-5, (p.per_layer_rel[0], expect)
    assert p.per_layer_rel[0] > 0


def test_legacy_trace_reproduction_is_deterministic_and_correctly_shaped(tiny):
    from mixed_study import legacy_trace as LT
    r1 = LT.reproduce_lyapunov(tiny, "the quick brown fox jumps over the lazy dog", seed=7, seq_len=16)
    r2 = LT.reproduce_lyapunov(tiny, "the quick brown fox jumps over the lazy dog", seed=7, seq_len=16)
    assert r1["rho_per_layer"] == r2["rho_per_layer"]            # same seed -> identical
    assert r1["n_blocks"] == tiny.n_layers and r1["n_rho"] == tiny.n_layers - 1
    r3 = LT.reproduce_lyapunov(tiny, "the quick brown fox jumps over the lazy dog", seed=8, seq_len=16)
    assert r1["rho_per_layer"] != r3["rho_per_layer"]            # different draw -> different profile
    assert LT.site_rho([0.5, 0.6], layer=1, mapping="into") == 0.5
    assert LT.site_rho([0.5, 0.6], layer=1, mapping="out") == 0.6
    assert LT.site_rho([0.5, 0.6], layer=0, mapping="into") is None


def test_directional_ladder_runs_and_validates_first_order(tiny):
    from mixed_study import directional_ladder as DL
    from quantbias.data import Example
    from quantbias.quantization import Quantizer
    q = Quantizer(tiny)
    ex = [Example(f"u{i}", "bbq", f"c{i}", "final", "g", f"Question {i}: pick one.\nAnswer:",
                  [" yes", " no", " maybe"], i % 3, meta={"context_condition": "disambig"}) for i in range(6)]
    res = DL.run(tiny, q, ex, layers=[0, 1, 2], bits=3, tag="tiny")
    assert set(res["sites"]) == {"L0.all", "L1.all", "L2.all"}
    for s in res["sites"].values():
        assert s["n"] == 6 and 0 <= s["pred_flip_rate"] <= 1 and 0 <= s["obs_flip_rate"] <= 1
        assert all(("pred_delta_s" in r and "actual_delta_s" in r) for r in s["rows"])
    assert "site_level" in res and res["site_level"]["n_sites"] == 3
    assert all(q.is_restored(c.id) for c in tiny.components)


def test_power_model_is_net_zero_under_pure_churn():
    import numpy as np
    from mixed_study.power import paired_power
    rng = np.random.default_rng(0)
    n = 600; dc = np.ones(n, bool); harm = rng.random(n) < 0.06
    clusters = [f"c{i // 4}" for i in range(n)]
    # zero true reduction, heavy churn -> power should be near alpha, not 0 and not 1
    p = paired_power(dc, harm, clusters, relative_reduction=0.0, churn=0.6, n_sim=60, n_boot=100)
    assert p < 0.25, p
