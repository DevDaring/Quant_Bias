import torch
from quantbias.data import Example
from quantbias.trace import capture_hidden, compare_captures, single_site_trace, propagation_features
from quantbias.quantization import Quantizer


def _examples():
    return [Example(f"u{i}", "winobias", "c", "selection", "female" if i % 2 else "male",
                    f"The nurse {i} said that", [" she", " he"], i % 2) for i in range(6)]


def test_identical_captures_have_zero_energy(tiny_adapter):
    ex = _examples()
    c = capture_hidden(tiny_adapter, ex, batch_size=4)
    assert c.hidden.shape == (6, tiny_adapter.n_layers + 1, 64)
    cmp = compare_captures(c, c, [e.label for e in ex])
    assert all(v == 0 for g in cmp["V"] for v in cmp["V"][g]) and cmp["lemma_violations"] == 0
    assert all(f == 0.0 for f in cmp["flip_rate"].values())


def test_single_site_injects_after_its_layer(tiny_adapter):
    a = tiny_adapter
    q = Quantizer(a)
    ex = _examples()
    dense = capture_hidden(a, ex, batch_size=4)
    cid = [c.id for c in a.components_of(1, "mlp")][0]      # layer 1 -> energy zero at layers 0..1 embeddings/block0
    tr = single_site_trace(a, q, cid, 3, ex, dense_capture=dense, batch_size=4)
    for g in tr.groups:
        assert tr.V[g][0] == 0 and tr.V[g][1] == 0 and tr.V[g][2] > 0
        assert tr.rho[g][1] is None       # undefined: no energy before injection
    assert tr.lemma_violations == 0
    f = propagation_features(tr)
    assert set(f) == {"male", "female"} and all("logit_linf" in v for v in f.values())
    assert all(q.is_restored(c.id) for c in a.components)
