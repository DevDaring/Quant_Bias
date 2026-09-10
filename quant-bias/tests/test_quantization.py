import torch
import pytest
from quantbias.quantization import (quantize_rtn, quantization_grid_check, gptq_quantize, PrecisionMap,
                               Quantizer, account_bytes, ByteFormat, component_bytes)


def test_rtn_grid_and_error():
    torch.manual_seed(0)
    w = torch.randn(32, 300)
    for bits in (3, 4, 8):
        deq = quantize_rtn(w, bits, group_size=128)
        chk = quantization_grid_check(w, deq, bits, 128)
        assert chk["grid_ok"], chk
        assert deq.shape == w.shape
    e4 = (w - quantize_rtn(w, 4)).abs().mean()
    e8 = (w - quantize_rtn(w, 8)).abs().mean()
    assert e8 < e4 / 8


def test_gptq_beats_rtn_on_correlated_inputs():
    torch.manual_seed(0)
    out_f, in_f, n = 16, 64, 2048
    W = torch.randn(out_f, in_f)
    basis = torch.randn(in_f, 8)
    X = torch.randn(n, 8) @ basis.t() + 0.05 * torch.randn(n, in_f)   # low-rank, correlated
    H = 2 * X.t() @ X / n
    Q_rtn = quantize_rtn(W, 3, group_size=32)
    Q_gptq = gptq_quantize(W, H, 3, group_size=32, blocksize=16)
    err_rtn = ((X @ W.t()) - (X @ Q_rtn.t())).pow(2).mean()
    err_gptq = ((X @ W.t()) - (X @ Q_gptq.t())).pow(2).mean()
    assert err_gptq < err_rtn
    assert quantization_grid_check(W, Q_gptq, 3, 32)["grid_ok"]


def test_bytes_monotone_and_breakdown(tiny_adapter):
    a = tiny_adapter
    b16 = account_bytes(a, PrecisionMap.dense(a))
    b8 = account_bytes(a, PrecisionMap.uniform(a, 8))
    b4 = account_bytes(a, PrecisionMap.uniform(a, 4))
    assert b16.total > b8.total > b4.total
    assert b4.metadata > 0 and b16.metadata == 0
    c = a.components[0]
    w, m, p = component_bytes(c, 4, ByteFormat())
    assert w == c.n_weights * 4 // 8
    # tied lm_head is counted once
    names = [e.name for e in a.excluded_tensors() if e.tied_to is None]
    assert not any("lm_head" in n for n in names) or not a.tied


def test_apply_and_exact_restore(tiny_adapter):
    a = tiny_adapter
    q = Quantizer(a)
    ids = a.tokenizer("hello world again", return_tensors="pt")
    with torch.no_grad():
        ref = a.model(**ids).logits.clone()
    pm = PrecisionMap.single_site(a, [a.components[0].id, a.components[-1].id], 4)
    q.apply_rtn(pm)
    assert q.current.bits_of(a.components[0].id) == 4
    assert q.current.bits_of(a.components[1].id) == 16
    with torch.no_grad():
        changed = a.model(**ids).logits
    assert not torch.allclose(ref, changed)
    q.restore()
    with torch.no_grad():
        back = a.model(**ids).logits
    assert torch.equal(ref, back)
    assert all(q.is_restored(c.id) for c in a.components)


def test_gptq_sequential_runs_on_conv1d_and_linear(tiny_adapter, tiny_llama_adapter):
    for a in (tiny_adapter, tiny_llama_adapter):
        q = Quantizer(a, group_size=32, blocksize=16)
        tok = a.tokenizer
        batches = [tok(t, return_tensors="pt", padding="max_length", max_length=16, truncation=True)
                   for t in ["the cat sat on the mat", "a nurse and a doctor", "numbers one two three four"]]
        pm = PrecisionMap.uniform(a, 4)
        q.apply_gptq(pm, batches, progress=None)
        s = q.stats_summary()
        assert s["n_quantized"] == len(a.components) and s["grid_ok_all"]
        q.restore()
        assert all(q.is_restored(c.id) for c in a.components)


def test_awq_is_explicit_error(tiny_adapter):
    q = Quantizer(tiny_adapter)
    with pytest.raises(NotImplementedError):
        q.apply(PrecisionMap.uniform(tiny_adapter, 4), "awq")
