import torch
from quantbias.model_adapters import check_weight_roundtrip


def test_conv1d_orientation_and_roundtrip(tiny_adapter):
    a = tiny_adapter
    assert a.family == "gpt2"
    c = a.component_by_id["L0.attn.c_attn"]
    assert c.is_conv1d and (c.out_features, c.in_features) == (192, 64)
    r = check_weight_roundtrip(a)
    assert r["ok"] and r["max_abs_logit_diff"] == 0.0
    # writing a transposed-orientation error must be caught
    w = a.get_weight("L0.mlp.c_fc")          # (256, 64)
    assert w.shape == (256, 64)
    import pytest
    with pytest.raises(ValueError):
        a.set_weight("L0.mlp.c_fc", torch.zeros(64, 300))


def test_llama_family_components(tiny_llama_adapter):
    a = tiny_llama_adapter
    assert a.family == "llama" and a.n_layers == 2
    names = [c.name for c in a.components_of(0)]
    assert names[:4] == ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
    assert {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"} <= set(names)
    assert not a.tied


def test_catcher_captures_layer0_inputs(tiny_llama_adapter):
    a = tiny_llama_adapter
    b = a.tokenizer(["one two three", "four five"], return_tensors="pt", padding=True)
    caught = a.catch_layer_inputs([b], layer=0)
    assert len(caught) == 1
    args, kw = caught[0]
    assert args[0].shape[-1] == 64
    assert a.layers[0] is a.model.model.layers[0]   # swapped back


def test_lfm2_hybrid_stack(tiny_lfm2_adapter):
    """Conv and attention blocks expose different components; both are found."""
    a = tiny_lfm2_adapter
    assert a.family == "lfm2" and a.n_layers == 4
    kinds = a.summary()["component_kinds"]
    assert kinds["attn"] > 0 and kinds["mlp"] > 0
    # a conv block and an attention block must not expose identical component sets
    assert {c.name for c in a.components_of(0)} != {c.name for c in a.components_of(1)}
    r = check_weight_roundtrip(a)
    assert r["ok"], r


def test_multimodal_decoder_is_located(tiny_multimodal_adapter):
    """The decoder nested under model.language_model is found; vision tower is not."""
    a = tiny_multimodal_adapter
    assert a.n_layers == 4
    assert not any("visual" in c.id or "vision" in c.id for c in a.components)
    r = check_weight_roundtrip(a)
    assert r["ok"], r
