"""Offline ReFactX constrained generation with vLLM.

Runs the same constrained decoding as the HuggingFace reference path
(``refactx/api.py``), but through vLLM's batched engine.

Prerequisites::

    pip install "refactx[vllm]"     # vllm>=0.12,<=0.18 (TRL-compatible)
    export REFACTX_INDEX=/path/to/index.txt   # or postgresql://... / http://...

Run::

    python examples/vllm_offline.py --model <hf-model-id> \
        --question "Who painted the Mona Lisa?"

Notes
-----
* **Thinking models** (Qwen3 / Qwen3.5, etc.): keep the default ``--no-thinking``.
  Their ``<think>`` ramble does not follow the ReFactX few-shot protocol and
  burns the token budget before any ``Fact:`` is emitted. ``--no-thinking``
  passes ``enable_thinking=False`` to the chat template; the model goes straight
  to ``Reasoning -> Fact: -> Answer:``.
* **Pre-Ampere GPUs** (Turing / T4, compute capability < 8.0): they have no
  FlashAttention-2, and vLLM's FlashInfer fallback JIT-compiles a kernel with
  ``nvcc`` -- which fails on boxes without the CUDA toolkit. This script defaults
  ``--attention-backend`` to ``TRITON_ATTN`` on such GPUs (Triton compiles via
  its bundled ptxas, no nvcc needed). Override with ``--attention-backend`` or
  pass ``auto`` to let vLLM choose.

Facts are recovered two ways: inline (parse ``Fact:`` spans from the text) and,
when a ``refactx_id`` is supplied, structurally via ``processor.pop_facts``.
"""
import argparse
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import refactx
from refactx import apply_prompt_template, PROMPT_TEMPLATE
from refactx.vllm_proc import RefactxLogitsProcessor


def _default_attention_backend():
    """Turing/T4 (sm<8.0) lack FA2; FlashInfer's nvcc JIT fails without a CUDA
    toolkit. Prefer the no-nvcc Triton backend there; let vLLM auto-pick elsewhere."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
            return "TRITON_ATTN"
    except Exception:
        pass
    return "auto"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id / path")
    ap.add_argument("--question", default="Who painted the Mona Lisa?")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="Cap context length (avoids KV-cache OOM on small GPUs).")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--attention-backend", default=_default_attention_backend(),
                    help="vLLM attention backend, or 'auto'. Default TRITON_ATTN on "
                         "pre-Ampere GPUs (no nvcc needed), 'auto' otherwise.")
    thinking = ap.add_mutually_exclusive_group()
    thinking.add_argument("--no-thinking", dest="thinking", action="store_false",
                          help="(default) disable <think> for thinking models.")
    thinking.add_argument("--thinking", dest="thinking", action="store_true",
                          help="keep the model's native thinking behaviour.")
    ap.set_defaults(thinking=False)
    args = ap.parse_args()

    if not os.environ.get("REFACTX_INDEX"):
        raise SystemExit("Set REFACTX_INDEX to the index location first.")

    # Build the few-shot prompt that teaches the model to emit `Fact:` spans.
    # enable_thinking is forwarded to the chat template; non-thinking models
    # ignore it harmlessly.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = apply_prompt_template(tokenizer, PROMPT_TEMPLATE, args.question,
                                   enable_thinking=args.thinking)

    # The processor is constructed by vLLM (it reads REFACTX_INDEX itself).
    llm_kwargs = dict(
        model=args.model,
        logits_processors=[RefactxLogitsProcessor],
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.attention_backend and args.attention_backend.lower() != "auto":
        llm_kwargs["attention_backend"] = args.attention_backend
    llm = LLM(**llm_kwargs)

    corr_id = "q0"
    params = SamplingParams(
        temperature=0.0,            # greedy == reference NUM_BEAMS=1, do_sample=False
        max_tokens=args.max_tokens,
        extra_args={"refactx": True, "refactx_id": corr_id},
    )

    out = llm.generate([prompt], params)
    text = out[0].outputs[0].text
    print("=== completion ===")
    print(text)

    # Structured facts (requires refactx_id). The processor instance lives in the
    # engine; reach it through the engine's logits-processor registry.
    print("\n=== facts (inline) ===")
    for line in text.splitlines():
        if line.strip().startswith("Fact:"):
            print(line.strip())


if __name__ == "__main__":
    main()
