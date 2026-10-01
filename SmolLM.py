import numpy as np
import torch.nn as nn
from dataclasses import dataclass
import torch
import math
from torch.nn import functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, start_pos=0):
        T = x.size(-2)
        pos = torch.arange(start_pos, start_pos + T, device=x.device, dtype=self.inv_freq.dtype).unsqueeze(1)  # [T,1]
        freqs = torch.matmul(pos, self.inv_freq.unsqueeze(0))  # [T, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, D]

        cos = emb.cos().unsqueeze(0).unsqueeze(0)  # [1,1,T,D]
        sin = emb.sin().unsqueeze(0).unsqueeze(0)

        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)

        return x * cos + rotated * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size // config.num_kv_heads, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size // config.num_kv_heads, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.n_head = config.num_atten_heads
        self.n_kv_head = config.num_kv_heads
        self.rotary = RotaryEmbedding(config.hidden_size // self.n_head, base=config.rope_theta)
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x, kv_cache=None, start_pos=0):
        B, T, C = x.size() # batch size, seq length, embedding dim
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_kv_head, C // self.n_head).transpose(1, 2) # (B, nh/3, T, hs)
        v = v.view(B, T, self.n_kv_head, C // self.n_head).transpose(1, 2) # (B, nh/3, T, hs)

        q = self.rotary(q, start_pos)
        k = self.rotary(k, start_pos)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_kv_cache = (k, v)

        repeats = self.n_head // self.n_kv_head
        k_rep = k.repeat_interleave(repeats, dim=1)
        v_rep = v.repeat_interleave(repeats, dim=1)

        att = (q @ k_rep.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
        att = att.masked_fill(self.causal_mask[:,:,start_pos:start_pos+T,:start_pos+T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        out = att @ v_rep
        out = out.transpose(1, 2).contiguous().view(B, T, C) # assembling heads
        out = self.o_proj(out)
        return out, new_kv_cache


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.swish = nn.SiLU()

    def forward(self, x):
        x = self.down_proj(self.up_proj(x) * self.swish(self.gate_proj(x)))
        return x


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, kv_cache=None, start_pos=0):
        attn_out, new_kv_cache = self.self_attn(self.input_layernorm(x), kv_cache, start_pos)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv_cache


@dataclass
class SmolLMConfig:
    vocab_size: int = 49152
    hidden_size: int = 576
    num_layers: int = 30
    num_atten_heads: int = 9
    num_kv_heads: int = 3
    intermediate_size: int = 1536
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    tie_word_embeddings: bool = True
    block_size: int = 1024


class SmolLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size),
            layers = nn.ModuleList([Layer(config) for _ in range(config.num_layers)]),
            norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, idx, kv_caches=None, start_pos=0):
        B, T = idx.size()
        assert start_pos + T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        x = self.model.embed_tokens(idx)

        new_kv_caches = []
        for i, layer in enumerate(self.model.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, layer_cache, start_pos)
            new_kv_caches.append(new_cache)

        x = self.model.norm(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        return logits, new_kv_caches

    @classmethod
    def from_pretrained(cls, checkpoint="HuggingFaceTB/SmolLM-135M-Instruct", device="cpu"):
        from transformers import AutoModelForCausalLM

        config_args = dict(vocab_size=49152, hidden_size=576, num_layers=30, num_atten_heads=9, num_kv_heads=3, intermediate_size=1536, rms_norm_eps=1e-5, rope_theta=10_000.0, tie_word_embeddings=True, block_size=1024)
        config = SmolLMConfig(**config_args)
        model = SmolLM(config)
        sd = model.state_dict()
        sd_keys = sd.keys()

        model_hf = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        model.load_state_dict(sd_hf, strict=False)

        return model


def top_p_sampling(logits, p=0.9, temperature=1.0):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # keep tokens until cumulative probability exceeds p
    keep_mask = cumulative_probs <= p
    if not keep_mask.any():
        keep_mask[0] = True
    else:
        keep_count = keep_mask.sum().item()
        keep_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
        keep_mask[:keep_count + 1] = True

    filtered_probs = torch.where(keep_mask, sorted_probs, torch.zeros_like(sorted_probs))
    filtered_probs = filtered_probs / filtered_probs.sum()

    sampled_idx = torch.multinomial(filtered_probs, 1).item()
    return int(sorted_indices[sampled_idx].item())


def generate(prompt, device='cpu', max_new_tokens=100, p=0.9, temperature=1.0):
    """
            {
                        "role": "system",
                        "content": (
                            "You are a Codenames clue generator. "
                            "Return exactly one English word in lowercase. "
                            "No punctuation, no explanation, no extra text, and stop immediately after the word."
                        )
            },
            {
                "role": "user",
                "content": "Words: apple, banana, cherry \n Clue:"
            },
            {
                "role": "assistant",
                "content": "fruit"
            },
            {
                "role": "user",
                "content": "Words: dog, wolf, fox \n Clue:"
            },
            {
                "role": "assistant",
                "content": "canine"
            },"""
    
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    generated = inputs.clone()
    stop_token_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}

    kv_caches = None
    start_pos = 0
    next_input = inputs

    for _ in range(max_new_tokens):
        logits, kv_caches = model(next_input, kv_caches=kv_caches, start_pos=start_pos)
        logits = logits[:, -1, :]   # [B, V]

        next_tokens = []
        for row in logits:
            next_tokens.append(top_p_sampling(row, p=p, temperature=temperature))
        next_tokens = torch.tensor(next_tokens, device=generated.device)
        generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
        
        if next_tokens.item() in stop_token_ids:
            break

        next_input = next_tokens.unsqueeze(1)
        start_pos = generated.size(1) - 1

    print(tokenizer.decode(generated))


if __name__ == "__main__":
    prompt = """Explain current in circuit theory"""
    # prompt = "Words: car, vehicle, transport. Propose in a single word a clue:"

    """"prompt = "Game: Codenames
    Words: apple, banana, cherry -> Clue: fruit
    Words: dog, wolf, fox -> Clue: canine
    Words: car, vehicle, transport -> Clue:"
    """

    from transformers import AutoTokenizer
    checkpoint="HuggingFaceTB/SmolLM-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = SmolLM.from_pretrained(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.eval()
    model.to(device)

    generate(
        prompt=prompt,
        device=device,
        max_new_tokens=100,
        p=0.8,
        temperature=1.0
    )

