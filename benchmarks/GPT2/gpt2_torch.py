"""GPT-2 124M inference reference in PyTorch.

Mirrors the dalotia C++ benchmark (benchmarks/GPT2/gpt2.cpp): same
HuggingFace safetensors layout, same prompt, same expected first
predicted token (407). Optimized for GPU inference and exportable to
TorchScript.

Inference optimizations:
  * inference_mode + eval (no autograd graph, no dropout)
  * fused F.scaled_dot_product_attention (Flash / mem-efficient on Ampere+)
  * KV cache for incremental decoding (O(S) per step instead of O(S^2))
  * optional autocast to bf16/fp16 on CUDA
  * optional torch.compile (Inductor) for kernel fusion
  * TF32 matmul enabled on CUDA
  * TorchScript export via torch.jit.script

Usage:
    python gpt2_torch.py [model.safetensors] [num_generate] \
        [--device cuda] [--dtype bf16] [--compile] \
        [--export gpt2_124m.ts]
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


# ── Hyperparameters (GPT-2 124M) ────────────────────────────────────────
@dataclass
class GPT2Config:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50257
    n_positions: int = 1024
    layer_norm_eps: float = 1e-5


# ── Model ───────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_embd = cfg.n_embd
        # Conv1D-style fused QKV (matches HuggingFace layout: weight is [in, 3*out])
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=True)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, S, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # [B, n_head, S, head_dim]
        q = q.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_head, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_cache = (k, v)

        # Fused SDPA — picks Flash / mem-efficient kernel on GPU.
        # is_causal only valid when S_q == S_k (prefill). For decode steps
        # (S_q==1, S_k>1) every query attends to all keys — no masking needed.
        is_causal = q.size(2) == k.size(2) and q.size(2) > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, S, C)
        return self.c_proj(y), new_cache


class MLP(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=True)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tanh-approximate GELU — matches the C++ benchmark's formula.
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        a, new_cache = self.attn(self.ln_1(x), kv_cache)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPT2(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        # Tied output projection (weight = wte.weight).

    def forward(
        self,
        tokens: torch.Tensor,                       # [B, S] int64
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S = tokens.shape
        pos = torch.arange(position_offset, position_offset + S,
                           device=tokens.device, dtype=torch.long)
        x = self.wte(tokens) + self.wpe(pos)[None, :, :]

        new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, blk in enumerate(self.h):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, nc = blk(x, cache_i)
            new_caches.append(nc)

        x = self.ln_f(x)
        # Tied lm_head: logits = x @ wte.weight^T
        logits = F.linear(x, self.wte.weight)
        return logits, new_caches


# ── Weight loading ──────────────────────────────────────────────────────
def load_hf_safetensors(model: GPT2, path: str) -> None:
    """Load HuggingFace gpt2 safetensors into our model.

    HF Conv1D stores weight as [in, out]; nn.Linear expects [out, in],
    so c_attn / c_proj / c_fc / mlp.c_proj weights must be transposed.
    """
    raw = load_file(path)
    sd = {}
    sd["wte.weight"] = raw["wte.weight"]
    sd["wpe.weight"] = raw["wpe.weight"]
    sd["ln_f.weight"] = raw["ln_f.weight"]
    sd["ln_f.bias"] = raw["ln_f.bias"]

    for i in range(model.cfg.n_layer):
        p = f"h.{i}."
        for sub in ("ln_1", "ln_2"):
            sd[p + f"{sub}.weight"] = raw[p + f"{sub}.weight"]
            sd[p + f"{sub}.bias"]   = raw[p + f"{sub}.bias"]
        for k in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            sd[p + f"{k}.weight"] = raw[p + f"{k}.weight"].t().contiguous()
            sd[p + f"{k}.bias"]   = raw[p + f"{k}.bias"]

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected}")


# ── Generation ──────────────────────────────────────────────────────────
@torch.inference_mode()
def generate(
    model: GPT2,
    prompt: List[int],
    num_new: int,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype] = None,
) -> List[int]:
    tokens = list(prompt)
    inp = torch.tensor([tokens], dtype=torch.long, device=device)

    def _run(x: torch.Tensor, caches, offset: int):
        # Required when compiling with CUDA Graphs (mode="reduce-overhead"):
        # signals that previous-step outputs (KV cache) won't be mutated.
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        if autocast_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                return model(x, caches, offset)
        return model(x, caches, offset)

    # Prefill
    logits, caches = _run(inp, None, 0)
    next_tok = int(logits[0, -1].argmax().item())
    tokens.append(next_tok)

    # Decode (one token at a time, KV cache reused)
    for _ in range(num_new - 1):
        x = torch.tensor([[next_tok]], dtype=torch.long, device=device)
        offset = len(tokens) - 1
        logits, caches = _run(x, caches, offset)
        next_tok = int(logits[0, -1].argmax().item())
        tokens.append(next_tok)

    return tokens


# ── TorchScript export ──────────────────────────────────────────────────
def export_torchscript(model: GPT2, out_path: str) -> None:
    """Script the model (preserves control flow) and save."""
    model.eval()
    scripted = torch.jit.script(model)
    scripted = torch.jit.optimize_for_inference(scripted)
    scripted.save(out_path)
    print(f"TorchScript model saved to {out_path}")


# ── Main ────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", nargs="?", default="./model.safetensors")
    ap.add_argument("num_generate", nargs="?", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
                    help="autocast dtype for inference (CUDA only)")
    ap.add_argument("--compile", action="store_true",
                    help="apply torch.compile (Inductor) for kernel fusion")
    ap.add_argument("--export", metavar="PATH", default=None,
                    help="export TorchScript model to PATH and exit")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    autocast_dtype = {"fp32": None,
                      "fp16": torch.float16,
                      "bf16": torch.bfloat16}[args.dtype]

    print(f"Loading GPT-2 124M (PyTorch/{device}) from {args.model_path} ...")
    cfg = GPT2Config()
    model = GPT2(cfg).to(device).eval()

    t0 = time.perf_counter()
    load_hf_safetensors(model, args.model_path)
    model.to(device)
    t1 = time.perf_counter()
    print(f"Model loaded in {t1 - t0:.3f}s")

    if args.export is not None:
        export_torchscript(model.to("cpu"), args.export)
        return 0

    if args.compile:
        # NB: mode="reduce-overhead" enables CUDA Graphs, which break KV-cache
        # reuse (returned tensors get overwritten on the next replay). Default
        # mode still gives Inductor kernel fusion without that hazard.
        model = torch.compile(model, fullgraph=False)

    # Same prompt as gpt2.cpp: "The meaning of life is" → expects 407 next.
    prompt_tokens = [464, 3616, 286, 1204, 318]

    print(f"Generating {args.num_generate} tokens...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    tokens = generate(model, prompt_tokens, args.num_generate, device, autocast_dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t3 = time.perf_counter()

    dt = t3 - t2
    print(f"Inference time: {dt:.3f}s ({args.num_generate} tokens, "
          f"{dt / args.num_generate:.4f}s/token)")
    print(f"Generated token IDs: {tokens}")

    # Validation: prefill of just the prompt should predict 407 next.
    with torch.inference_mode():
        inp = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        if autocast_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits, _ = model(inp, None, 0)
        else:
            logits, _ = model(inp, None, 0)
    if not torch.isfinite(logits).all():
        print("FAIL: non-finite logits!"); return 1
    first_pred = int(logits[0, -1].argmax().item())
    print(f"First predicted token after prompt: {first_pred}")
    if first_pred != 407:
        print(f"FAIL: expected token 407, got {first_pred}")
        return 1

    print("success!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
