import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Codes/quant-bias


@pytest.fixture(scope="session")
def tiny_adapter():
    """Random-init 2-layer GPT-2 (Conv1D storage) -- no download, fast."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
    from quantbias.model_adapters import ModelAdapter
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=2, n_embd=64, n_head=4, n_positions=128, vocab_size=50257)
    model = GPT2LMHeadModel(cfg).eval()
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tok.pad_token = tok.eos_token
    return ModelAdapter(model, tok, "tiny-gpt2", "test")


@pytest.fixture(scope="session")
def tiny_llama_adapter():
    """Random-init 2-layer Llama (nn.Linear storage, tied head off)."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
    from quantbias.model_adapters import ModelAdapter
    torch.manual_seed(0)
    cfg = LlamaConfig(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=50257, max_position_embeddings=128, tie_word_embeddings=False)
    model = LlamaForCausalLM(cfg).eval()
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tok.pad_token = tok.eos_token
    return ModelAdapter(model, tok, "tiny-llama", "test")


@pytest.fixture(scope="session")
def tiny_lfm2_adapter():
    """Random-init hybrid conv/attention stack in the shape of LFM2-2.6B."""
    import torch
    from transformers import Lfm2Config, Lfm2ForCausalLM, AutoTokenizer
    from quantbias.model_adapters import ModelAdapter
    torch.manual_seed(0)
    cfg = Lfm2Config(num_hidden_layers=4, hidden_size=64, intermediate_size=128,
                     num_attention_heads=4, num_key_value_heads=2, vocab_size=50257,
                     block_dim=64, conv_dim=64, block_ff_dim=128,
                     layer_types=["conv", "full_attention", "conv", "full_attention"])
    model = Lfm2ForCausalLM(cfg).eval()
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tok.pad_token = tok.eos_token
    return ModelAdapter(model, tok, "tiny-lfm2", "test")


@pytest.fixture(scope="session")
def tiny_multimodal_adapter():
    """Random-init VLM wrapper whose decoder is nested under model.language_model."""
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, AutoTokenizer
    from quantbias.model_adapters import ModelAdapter
    torch.manual_seed(0)
    cfg = Qwen3_5Config(
        text_config=dict(num_hidden_layers=4, hidden_size=64, intermediate_size=128,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         vocab_size=50257, layer_types=["linear_attention", "full_attention"] * 2,
                         linear_key_head_dim=16, linear_value_head_dim=16,
                         linear_num_key_heads=2, linear_num_value_heads=2,
                         linear_conv_kernel_dim=4, tie_word_embeddings=True),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=64, patch_size=16))
    model = Qwen3_5ForConditionalGeneration(cfg).eval()
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tok.pad_token = tok.eos_token
    return ModelAdapter(model, tok, "tiny-qwen35", "test")
