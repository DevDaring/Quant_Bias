import torch
import pytest
from quantbias.baselines import (contrastive_hessian, sparsegpt_prune, Comparators,
                                 _last_token_inputs)
from quantbias.data import Example, Pair
from quantbias.quantization import Quantizer, PrecisionMap


def _pairs(n=6):
    out = []
    for i in range(n):
        a = Example(f"a{i}", "winobias", f"c{i}", "calibration", "female",
                    f"The nurse number {i} said that", [" she", " he"], 0, meta={"condition": "pro"})
        b = Example(f"b{i}", "winobias", f"c{i}", "calibration", "male",
                    f"The doctor number {i} said that", [" she", " he"], 0, meta={"condition": "anti"})
        out.append(Pair(f"p{i}", "winobias", a, b, "pronoun"))
    return out


def test_contrastive_hessian_is_psd_and_captures_difference():
    torch.manual_seed(0)
    A = torch.randn(20, 8)
    H = contrastive_hessian(A, A.clone())
    assert torch.allclose(H, torch.zeros_like(H), atol=1e-6)   # identical groups -> no term
    B = A + torch.randn(20, 8)
    H2 = contrastive_hessian(A, B)
    assert torch.linalg.eigvalsh(H2).min() >= -1e-5            # PSD, so Cholesky stays valid
    assert H2.abs().sum() > 0


def test_sparsegpt_hits_target_sparsity_and_beats_magnitude():
    torch.manual_seed(0)
    W = torch.randn(16, 64)
    X = torch.randn(512, 64) @ torch.randn(64, 64) * 0.3
    H = 2 * X.t() @ X / X.shape[0]
    P = sparsegpt_prune(W, H, 0.5)
    frac = (P == 0).float().mean().item()
    assert 0.42 <= frac <= 0.58, frac
    mag = W * (W.abs() >= W.abs().flatten().kthvalue(int(W.numel() * 0.5)).values)
    err_sg = ((X @ W.t()) - (X @ P.t())).pow(2).mean()
    err_mag = ((X @ W.t()) - (X @ mag.t())).pow(2).mean()
    assert err_sg < err_mag


def test_sparsegpt_semi_structured_2of4():
    torch.manual_seed(0)
    W = torch.randn(8, 32)
    H = torch.eye(32) * 2
    P = sparsegpt_prune(W, H, 0.5, prune_n=2, prune_m=4)
    for r in range(P.shape[0]):
        for blk in range(0, 32, 4):
            assert int((P[r, blk:blk + 4] == 0).sum()) == 2


def test_comparators_run_and_restore(tiny_adapter):
    a = tiny_adapter
    q = Quantizer(a, group_size=32, blocksize=16)
    c = Comparators(a, q, batch_size=3)
    tok = a.tokenizer
    batches = [tok(t, return_tensors="pt", padding="max_length", max_length=16, truncation=True)
               for t in ["the cat sat", "a nurse spoke", "one two three"]]
    ids = a.tokenizer("hello there", return_tensors="pt")
    with torch.no_grad():
        ref = a.model(**ids).logits.clone()
    pairs = _pairs()

    for build in (lambda: c.fair_gptq(4, batches, pairs, 1.0, 6),
                  lambda: c.critical_weight_protection(4, pairs, 0.02, 6),
                  lambda: c.sparsegpt(0.5, batches),
                  lambda: c.debias_sparsegpt(0.5, batches, pairs, 1.0, 6)):
        br = build()
        assert br.reimplementation and br.assumptions
        with torch.no_grad():
            changed = a.model(**ids).logits
        assert not torch.allclose(ref, changed), br.name
        q.restore()
        with torch.no_grad():
            assert torch.equal(ref, a.model(**ids).logits), f"{br.name} did not restore"


def test_cwp_charges_its_index_overlay(tiny_adapter):
    a = tiny_adapter
    q = Quantizer(a)
    c = Comparators(a, q, batch_size=3)
    br = c.critical_weight_protection(4, _pairs(), protect_frac=0.02, max_pairs=6)
    q.restore()
    assert br.extra["n_protected"] > 0
    # 48 bits per protected weight must actually appear in the total
    assert br.bytes_total == br.extra["base_bytes"] + br.extra["overlay_bytes"]
    assert br.extra["overlay_bytes"] == br.extra["n_protected"] * 48 // 8
