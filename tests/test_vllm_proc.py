"""CPU-only tests for the vLLM logits-processor adapter (refactx/vllm_proc.py).

These exercise the *new* plumbing -- persistent-batch bookkeeping in
``update_state`` and the masking in ``apply`` -- without needing a GPU or a real
vLLM engine. The constrained-decoding core itself is reused verbatim from
``refactx.generate`` and is covered by the existing trie tests.

Requires ``vllm`` (for the base class + ``BatchUpdate``), ``torch`` and
``transformers`` to be installed; skipped otherwise.

Run::  pytest tests/test_vllm_proc.py -q
"""
import math
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("transformers")

from transformers import AutoTokenizer  # noqa: E402
from vllm.v1.sample.logits_processor import BatchUpdate, MoveDirectionality  # noqa: E402

from refactx.generate import ConstrainedLogitsProcessor  # noqa: E402
from refactx.vllm_proc import RefactxLogitsProcessor, _ReqState  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


class FakeIndex:
    """Trie stub: maps a partial-fact token tuple -> {allowed_token: leafcount}."""

    def __init__(self, allowed):
        self.allowed = allowed

    def next_tokens(self, sequence, state=None):
        return dict(self.allowed.get(tuple(sequence), {})), None


def make_proc(tokenizer, index):
    """Build the processor without going through vLLM's engine __init__."""
    proc = RefactxLogitsProcessor.__new__(RefactxLogitsProcessor)
    proc.device = torch.device("cpu")
    proc.tokenizer = tokenizer
    proc.index = index
    proc._cgen = ConstrainedLogitsProcessor(index=index, states=None, tokenizer=tokenizer)
    proc._req = {}
    proc._facts = {}
    return proc


def added(idx, enabled=True, corr=None, out=None):
    xa = {}
    if enabled:
        xa["refactx"] = True
    if corr is not None:
        xa["refactx_id"] = corr
    params = types.SimpleNamespace(extra_args=xa)
    return (idx, params, None, [] if out is None else out)


def bu(batch_size=4, removed=(), add=(), moved=()):
    return BatchUpdate(batch_size=batch_size, removed=list(removed),
                       added=list(add), moved=list(moved))


def test_only_enabled_requests_are_tracked(tokenizer):
    proc = make_proc(tokenizer, FakeIndex({}))
    proc.update_state(bu(add=[added(0, enabled=True), added(1, enabled=False)]))
    assert set(proc._req) == {0}


def test_removed_request_is_dropped(tokenizer):
    proc = make_proc(tokenizer, FakeIndex({}))
    proc.update_state(bu(add=[added(0), added(1)]))
    proc.update_state(bu(removed=[0]))
    assert set(proc._req) == {1}


def test_move_swap_and_unidirectional(tokenizer):
    proc = make_proc(tokenizer, FakeIndex({}))
    proc.update_state(bu(add=[added(0), added(1)]))
    s0, s1 = proc._req[0], proc._req[1]

    proc.update_state(bu(moved=[(0, 1, MoveDirectionality.SWAP)]))
    assert proc._req[0] is s1 and proc._req[1] is s0

    # Unidirectional move 1 -> 0 leaves slot 1 vacated.
    proc.update_state(bu(moved=[(1, 0, MoveDirectionality.UNIDIRECTIONAL)]))
    assert proc._req[0] is s0 and 1 not in proc._req


def test_apply_masks_to_allowed_tokens(tokenizer):
    # When constrained at the start of a fact, only the index's allowed tokens
    # survive; everything else is set to -inf.
    proc = make_proc(tokenizer, FakeIndex({(): {3: 1, 7: 1}}))
    proc.update_state(bu(add=[added(0)]))
    rs = proc._req[0]
    rs.state.state = rs.state.CONSTRAINED_GENERATION  # force constrained
    rs.state.token_ids = []

    logits = torch.zeros(1, 10)
    out = proc.apply(logits)

    allowed = {3, 7}
    for t in range(10):
        if t in allowed:
            assert out[0, t].item() == 0.0
        else:
            assert math.isinf(out[0, t].item()) and out[0, t].item() < 0


def test_apply_passthrough_when_not_constrained(tokenizer):
    proc = make_proc(tokenizer, FakeIndex({(): {3: 1}}))
    proc.update_state(bu(add=[added(0)]))
    # state stays NORMAL (no trigger seen)
    logits = torch.arange(10, dtype=torch.float32).reshape(1, 10).clone()
    out = proc.apply(logits)
    assert torch.equal(out, torch.arange(10, dtype=torch.float32).reshape(1, 10))


def test_trigger_switches_to_constrained(tokenizer):
    # Feeding generated tokens whose text ends with "Fact:" must flip the row
    # into constrained mode on the next apply.
    proc = make_proc(tokenizer, FakeIndex({(): {5: 1}}))
    out_ids = tokenizer.encode("The capital is Paris. Fact:")
    proc.update_state(bu(add=[added(0, out=out_ids)]))
    proc.apply(torch.zeros(1, tokenizer.vocab_size))
    assert proc._req[0].state.is_constrained()


def test_facts_stashed_for_correlation_id(tokenizer):
    proc = make_proc(tokenizer, FakeIndex({}))
    proc.update_state(bu(add=[added(0, corr="q0")]))
    # Simulate a completed fact (cache_add stores sequence[:-1]).
    triple_ids = tokenizer.encode(" Mona Lisa creator Leonardo")
    proc._req[0].state.generated_triples.append(triple_ids)
    proc.update_state(bu(removed=[0]))

    facts = proc.pop_facts("q0")
    assert len(facts) == 1 and "Leonardo" in facts[0]
    assert proc.pop_facts("q0") == []  # popped, so empty now
