import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mixed_study  # noqa: F401  registers ../quant-bias


@pytest.fixture(scope="session")
def tiny():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
    from quantbias.model_adapters import ModelAdapter
    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(n_layer=3, n_embd=64, n_head=4, n_positions=128, vocab_size=50257)).eval()
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2"); tok.pad_token = tok.eos_token
    return ModelAdapter(m, tok, "tiny", "test")


@pytest.fixture
def rows():
    from mixed_study.records import Row
    def mk(uid, bench, cluster, group, label, ls, cond=None, target=None, unk=None, ags=(), **meta):
        n = len(ls)
        pred = max(range(n), key=lambda i: ls[i])
        return Row(uid, bench, cluster, group, label, pred, (pred == label) if label is not None else None,
                   tuple(ls), tuple(x / 2 for x in ls), 0.0, cond, target, unk, "Age", tuple(ags), {"category": "Age"}, meta)
    return mk
