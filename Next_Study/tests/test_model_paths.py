"""Model-facing tests on GPT-2 Small (CPU): exact dense restoration after a
single-layer intervention, fp32 score accumulation, and the all-boundary gradient
being identical to the completed study's one-boundary-at-a-time gradient."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from next_study import synthbias as S
from next_study import directional as D
from next_study.common import DATA


@pytest.fixture(scope="module")
def adapter():
    try:
        from next_study.confirmation import load_adapter
        return load_adapter("DBG", device="cpu", attn=None, dtype="fp32")
    except Exception as e:                               # no cached checkpoint on this machine
        pytest.skip(f"GPT-2 not available: {e}")


@pytest.fixture(scope="module")
def examples(adapter):
    p = DATA / "synthbias_canonical.jsonl"
    if not p.exists():
        pytest.skip("run ingest first")
    rows = [r for r in S.type2(S.load_rows(p), "final")][:4]
    return S.build_examples(rows, "P3", "DBG", adapter.tokenizer, chat=False)


def test_exact_restoration_after_single_layer_intervention(adapter, examples):
    from next_study.confirmation import quantizer_for
    from quantbias.quantization import PrecisionMap
    q = quantizer_for(adapter)
    comp_ids = [c.id for c in adapter.components_of(3)]
    before = {c: adapter.get_weight(c).clone() for c in comp_ids}
    q.apply_rtn(PrecisionMap.single_site(adapter, comp_ids, 4), record_stats=False)
    assert any(not torch.equal(adapter.get_weight(c), before[c]) for c in comp_ids)
    q.restore()
    assert all(torch.equal(adapter.get_weight(c), before[c]) for c in comp_ids)
    assert all(q.is_restored(c) for c in comp_ids)


def test_scores_are_fp32_and_finite(adapter, examples):
    from quantbias.evaluate import score_candidates
    out = score_candidates(adapter, examples, batch_size=2, progress_every=0)
    for s in out:
        assert all(np.isfinite(v) for v in s.logprob_sum) and all(v <= 0 for v in s.logprob_sum)
    ids, start = D._encode(adapter, examples[0].prompt, examples[0].candidates[0])
    score, grads = D.all_boundary_gradients(adapter, ids, start)
    assert abs(score - out[0].logprob_sum[0]) < 1e-3          # same encoding as the scorer
    assert all(g.dtype == torch.float32 and g.shape == (start, adapter.model.config.n_embd) for g in grads)
    assert len(grads) == adapter.n_layers


def test_all_boundary_gradient_matches_per_boundary_gradient(adapter, examples):
    """The completed study computed grad at one boundary per backward pass; the new code
    takes every boundary from one pass. They must agree at every layer."""
    from mixed_study import directional as MD
    from mixed_study import residuals as R
    e = examples[0]
    g_all = D.gradient_phase(adapter, [e], log_every=0)[e.uid]
    for li in (0, adapter.n_layers // 2, adapter.n_layers - 1):
        cg = MD.contrast_and_gradient(adapter, e, R.Boundary(li))
        # the completed code encodes without BOS; GPT-2 adds none either way, so positions align
        old = cg["grad_contrast"].float()
        new = g_all["grad_contrast"][li]
        n = min(old.shape[0], new.shape[0])
        assert torch.allclose(old[-n:], new[-n:], atol=1e-4, rtol=1e-3), f"layer {li} gradient mismatch"
    assert abs(g_all["margin_wr"]) == pytest.approx(abs(g_all["margin_gold"]), abs=1e-6)


def test_layer_step_restores_and_records(adapter, examples):
    from next_study.confirmation import quantizer_for
    grads = D.gradient_phase(adapter, examples, log_every=0)
    res = D.layer_step(adapter, quantizer_for(adapter), examples, grads, li=2, bits=4, batch_size=2)
    s = res["summary"]
    assert s["restored_exact"] and s["n"] == len(examples)
    assert all(np.isfinite(r["pred_delta"]) and np.isfinite(r["actual_delta"]) for r in res["rows"])
    assert 0 <= s["pred_flip_rate"] <= 1 and 0 <= s["obs_flip_rate"] <= 1
