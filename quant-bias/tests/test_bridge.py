import json
from quantbias.bridge import align_with_sites, interpret, _layer_of, load_prior_rho


def test_layer_parsing():
    assert _layer_of("L12.mlp.up_proj") == 12
    assert _layer_of("L0.all") == 0
    assert _layer_of("weird") is None


def test_prior_rho_loads_for_known_models():
    r = load_prior_rho("mistral_7b")
    assert r and len(r["rho_per_layer"]) > 20 and r["source"]
    assert load_prior_rho("does_not_exist") is None


def _sites(n=12, coupled=True):
    out = {}
    for i in range(n):
        flip = 0.02 * i if coupled else 0.05
        out[f"L{i}.all"] = {
            "components": [f"L{i}.attn.c_attn"], "granularity": "layer",
            "features_selection": {"g": {"V_final": 1e-4 * (i + 1), "logit_linf": 0.1,
                                         "margin_dense": 1.0, "flip_rate": flip,
                                         "geo_mean_rho": 1.0, "max_rho": 1.0,
                                         "sys_frac_final": 0.1, "abs_drift_final": 1.0}},
            "observed_final": {"first_token_flip_rate": {"g": flip, "h": flip / 2},
                               "full_score_flips": {"harmful_flip_rate": flip}},
            "ppl_utility": 10.0 + i}
    return out


def test_bridge_aligns_and_correlates():
    out = align_with_sites("mistral_7b", _sites())
    assert out["prior_rho"] is not None
    rows = [r for r in out["rows"] if r["prior_rho"] is not None]
    assert rows, "no site matched a prior rho index"
    c = out["correlations"]
    assert "prior_rho_vs_harmful_flips" in c and "V_final_vs_harmful_flips" in c
    # this study's own energy is monotone in harm by construction
    assert c["V_final_vs_harmful_flips"]["spearman"] > 0.9
    assert isinstance(interpret(out), str) and len(interpret(out)) > 40


def test_bridge_handles_missing_prior():
    out = align_with_sites("llama_2_7b", _sites())
    assert out["prior_rho"] is None
    assert all(r["prior_rho"] is None for r in out["rows"])
    assert "no bridge claim" in interpret(out).lower()
