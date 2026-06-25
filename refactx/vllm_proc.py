"""vLLM V1 custom logits processor that runs ReFactX constrained generation.

ReFactX normally constrains generation through a HuggingFace ``LogitsProcessor``
(``refactx.generate.ConstrainedLogitsProcessor``). vLLM has its own logits
processor interface (``vllm.v1.sample.logits_processor.LogitsProcessor``) that is
*batch oriented*: a single instance sees the whole running batch every decode
step, is told about added/removed/moved requests via ``update_state``, and is
*not* handed the token sequences directly. This module bridges the two.

Design notes
------------
* The per-sequence state machine (detect the ``Fact:`` trigger, walk the prefix
  tree, dedup) is reused **verbatim** from ``refactx.generate``:
  :class:`~refactx.generate.PatternConstrainedState` for the state machine and
  :meth:`~refactx.generate.ConstrainedLogitsProcessor.constrained_generation`
  for the masking + duplicate-avoidance logic. This guarantees that constrained
  decoding behaves identically to the HuggingFace path.
* Beam search is intentionally **not** supported -- vLLM custom logits
  processors run under token-by-token sampling/greedy only. This matches the
  ReFactX reference server default (``NUM_BEAMS=1``).
* The processor only constrains requests that opt in via
  ``SamplingParams.extra_args={"refactx": True}``; all other requests pass
  through untouched, so a single served model can mix constrained and normal
  traffic.

Compatible with ``vllm>=0.12,<=0.18`` (the range TRL supports). The custom
logits-processor API used here (``BatchUpdate`` / ``validate_params`` /
``apply``) is stable across that range.

Usage (offline)::

    import os
    os.environ["REFACTX_INDEX"] = "/path/to/index.txt"   # or postgres://... / http://...
    from vllm import LLM, SamplingParams
    from refactx.vllm_proc import RefactxLogitsProcessor

    llm = LLM(model="...", logits_processors=[RefactxLogitsProcessor])
    out = llm.generate(prompts, SamplingParams(temperature=0,
                                               max_tokens=800,
                                               extra_args={"refactx": True}))

Usage (server)::

    REFACTX_INDEX=/path/to/index.txt \
        vllm serve <model> --logits-processors refactx.vllm_proc:RefactxLogitsProcessor
    # client opts in per request with extra_body={"vllm_xargs": {"refactx": true}}
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch

from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

import refactx
from refactx.generate import ConstrainedLogitsProcessor, PatternConstrainedState
from refactx.index import DictIndex

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm import SamplingParams
    from vllm.config import VllmConfig

# Key the caller sets in SamplingParams.extra_args to enable constrained decoding.
ENABLE_KEY = "refactx"
# Optional correlation id so callers can retrieve the structured facts afterwards.
CORR_ID_KEY = "refactx_id"
# Env var pointing at the ReFactX index (file path / postgresql:// / http(s)://).
INDEX_ENV = "REFACTX_INDEX"
# Trigger that switches a sequence from free to constrained generation.
TRIGGER_PATTERN = "Fact:"


def _extra_args(params) -> dict:
    return getattr(params, "extra_args", None) or {}


def _enabled(params) -> bool:
    return bool(_extra_args(params).get(ENABLE_KEY))


class _ReqState:
    """Per persistent-batch-row bookkeeping for one constrained request."""

    __slots__ = ("state", "out_ids", "consumed", "corr_id")

    def __init__(self, state: PatternConstrainedState, out_ids, corr_id):
        self.state = state
        # Live reference to the request's output-token list; vLLM appends to it
        # in place every decode step. We track how many we have already fed to
        # the state machine via ``consumed``.
        self.out_ids = out_ids
        self.consumed = 0
        self.corr_id = corr_id


class RefactxLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        self.device = device

        model_cfg = vllm_config.model_config
        tok_id = getattr(model_cfg, "tokenizer", None) or model_cfg.model
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_id,
            trust_remote_code=getattr(model_cfg, "trust_remote_code", False),
        )

        index_url = os.environ.get(INDEX_ENV)
        if not index_url:
            raise ValueError(
                f"{INDEX_ENV} is not set. Point it at the ReFactX index "
                f"(a text file, postgresql:// URL, or http(s):// endpoint)."
            )
        # The text-file backend tokenizes triples and needs the tokenizer; the
        # postgres/http backends build the trie remotely and take no tokenizer.
        if os.path.isfile(index_url):
            self.index = refactx.load_index(index_url, tokenizer=self.tokenizer)
        else:
            self.index = refactx.load_index(index_url)

        # Reuse the HF masking + duplicate-avoidance logic unchanged. ``states``
        # is unused inside ``constrained_generation`` (state is passed per call),
        # so ``None`` is fine.
        self._cgen = ConstrainedLogitsProcessor(
            index=self.index, states=None, tokenizer=self.tokenizer
        )

        # Active constrained requests, keyed by persistent-batch row index.
        self._req: dict[int, _ReqState] = {}
        # Completed requests' facts, keyed by caller correlation id.
        self._facts: dict[object, list[str]] = {}

    # -- vLLM LogitsProcessor interface ------------------------------------

    @classmethod
    def validate_params(cls, params: "SamplingParams") -> None:
        xa = getattr(params, "extra_args", None)
        if xa and ENABLE_KEY in xa and not isinstance(xa[ENABLE_KEY], bool):
            raise ValueError(
                f"extra_args['{ENABLE_KEY}'] must be a bool (got "
                f"{type(xa[ENABLE_KEY]).__name__})."
            )

    def is_argmax_invariant(self) -> bool:
        # Masking can change which token is the argmax, so this processor must
        # run even under greedy sampling.
        return False

    def update_state(self, batch_update: Optional["BatchUpdate"]) -> None:
        if batch_update is None:
            return

        # Order matches vLLM's built-in processors: removed, added, then moved.
        for index in batch_update.removed:
            self._finish(self._req.pop(index, None))

        for index, params, _prompt_ids, output_ids in batch_update.added:
            # A freed slot may be reused by a non-constrained request.
            self._finish(self._req.pop(index, None))
            if _enabled(params):
                self._req[index] = _ReqState(
                    self._new_state(),
                    output_ids,
                    _extra_args(params).get(CORR_ID_KEY),
                )

        for a_index, b_index, direct in batch_update.moved:
            a_entry = self._req.pop(a_index, None)
            b_entry = self._req.pop(b_index, None)
            if a_entry is not None:
                self._req[b_index] = a_entry
            if b_entry is not None and direct == MoveDirectionality.SWAP:
                self._req[a_index] = b_entry

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._req:
            return logits

        n_rows = logits.shape[0]
        mask: Optional[torch.Tensor] = None

        for i, rs in self._req.items():
            # Feed every output token generated since we last looked. In greedy/
            # sampling this is exactly one token per step, but draining the delta
            # is robust to chunked prefill or multi-token appends.
            out = rs.out_ids
            if rs.consumed < len(out):
                for tok in out[rs.consumed:]:
                    rs.state.update(tok)
                rs.consumed = len(out)

            if i >= n_rows:
                continue

            if rs.state.is_constrained():
                if mask is None:
                    mask = torch.zeros_like(logits)
                mask[i] = float("-inf")
                # During constrained mode ``token_ids`` holds exactly the tokens
                # generated since the trigger (it is cleared on state entry), so
                # it is the partial-fact sequence the trie is walked with.
                seq = list(rs.state.token_ids)
                self._cgen.constrained_generation(seq, mask, i, state=rs.state)

        if mask is not None:
            logits = logits + mask
        return logits

    # -- helpers / public extras -------------------------------------------

    def _new_state(self) -> PatternConstrainedState:
        return PatternConstrainedState(
            pattern=TRIGGER_PATTERN,
            tokenizer=self.tokenizer,
            cache_index=DictIndex(),
            subtree_cache=DictIndex(),
        )

    def _finish(self, rs: Optional[_ReqState]) -> None:
        """Stash a finished request's facts so callers can retrieve them."""
        if rs is None or rs.corr_id is None:
            return
        self._facts[rs.corr_id] = [
            self.tokenizer.decode(triple_ids, skip_special_tokens=True)
            for triple_ids in rs.state.generated_triples
        ]

    def pop_facts(self, corr_id) -> list[str]:
        """Return (and clear) the facts generated for a finished request.

        Only populated when the request was submitted with
        ``extra_args={"refactx_id": <corr_id>}``. Facts also appear inline in the
        completion text as ``Fact: ...`` spans, which is the simplest way to
        recover them on the OpenAI server path.
        """
        return self._facts.pop(corr_id, [])
