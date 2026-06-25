"""Offline ReFactX constrained generation with vLLM.

Runs the same constrained decoding as the HuggingFace reference path
(``refactx/api.py``), but through vLLM's batched engine.

Prerequisites::

    pip install "refactx[vllm]"     # vllm>=0.12,<=0.18 (TRL-compatible)
    export REFACTX_INDEX=/path/to/index.txt   # or postgresql://... / http://...

Run::

    python examples/vllm_offline.py --model <hf-model-id> \
        --question "Who painted the Mona Lisa?"

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id / path")
    ap.add_argument("--question", default="Who painted the Mona Lisa?")
    ap.add_argument("--max-tokens", type=int, default=800)
    args = ap.parse_args()

    if not os.environ.get("REFACTX_INDEX"):
        raise SystemExit("Set REFACTX_INDEX to the index location first.")

    # Build the few-shot prompt that teaches the model to emit `Fact:` spans.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = apply_prompt_template(tokenizer, PROMPT_TEMPLATE, args.question)

    # The processor is constructed by vLLM (it reads REFACTX_INDEX itself).
    llm = LLM(model=args.model, logits_processors=[RefactxLogitsProcessor])

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
