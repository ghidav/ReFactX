"""CPU-only tests for the decode-time sentinel in ConstrainedLogitsProcessor.

When ``avoid_duplicates`` is on, generating a triple removes its leaf, and once a
subject-relation has no leaves left ``subtract_tokens`` deletes the relation token
entirely -- so the model can no longer select that relation and is forced to
enumerate other relations of the subject. With ``sentinel=True`` the exhausted
relation stays selectable and its object slot yields a fixed "no further records"
object instead, giving the model an explicit "nothing more here" signal.

These drive ``constrained_generation`` directly along a known triple (after
marking it generated) -- no GPU, no vLLM engine, no large download.

Run::  pytest tests/test_sentinel.py -q
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import AutoTokenizer  # noqa: E402

from refactx.generate import ConstrainedLogitsProcessor, PatternConstrainedState  # noqa: E402
from refactx.index import DictIndex  # noqa: E402

# Subject <A> with two relations: <r> (the one we exhaust) and <s> (stays new).
TRIPLE = " <A> <r> <x> ."
OTHER = " <A> <s> <y> ."


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def index(tokenizer):
    idx = DictIndex()
    for t in (TRIPLE, OTHER):
        idx.add(tokenizer.encode(t, add_special_tokens=False), new_leaf=True)
    return idx


def _triple_ids(tokenizer):
    return tokenizer.encode(TRIPLE, add_special_tokens=False)


def _state(tokenizer):
    """A state in which TRIPLE has already been generated (so it's in the dedup cache)."""
    st = PatternConstrainedState("Fact:", tokenizer, DictIndex(), DictIndex())
    st.cache_index.add(list(_triple_ids(tokenizer)), new_leaf=True)
    return st


def _allowed(proc, state, prefix, vocab):
    """Token ids left unmasked after one constrained step on `prefix`."""
    mask = torch.full((1, vocab), float("-inf"))
    proc.constrained_generation(list(prefix), mask, 0, state)
    return set((mask[0] == 0).nonzero(as_tuple=True)[0].tolist())


def _object_slot(tokenizer, T):
    # Object starts after the second "> <" boundary of "<S> <R> <O>".
    return next(k for k in range(1, len(T) + 1)
               if tokenizer.decode(T[:k]).count("> <") >= 2)


def test_without_sentinel_exhausted_relation_is_pruned(tokenizer, index):
    """Baseline bug: the exhausted relation token gets deleted from the allowed set."""
    T = _triple_ids(tokenizer)
    V = len(tokenizer)
    obj = _object_slot(tokenizer, T)
    proc = ConstrainedLogitsProcessor(index, None, tokenizer=tokenizer, sentinel=False)
    pruned = [k for k in range(1, obj)
              if T[k] not in _allowed(proc, _state(tokenizer), T[:k], V)]
    assert pruned, "expected the exhausted relation to be pruned before the object slot"


def test_with_sentinel_relation_stays_selectable(tokenizer, index):
    """The whole subject-relation remains traversable up to the object slot."""
    T = _triple_ids(tokenizer)
    V = len(tokenizer)
    obj = _object_slot(tokenizer, T)
    proc = ConstrainedLogitsProcessor(index, None, tokenizer=tokenizer, sentinel=True)
    for k in range(1, obj):
        assert T[k] in _allowed(proc, _state(tokenizer), T[:k], V), \
            f"real token at position {k} should stay selectable with sentinel on"


def test_with_sentinel_object_slot_emits_sentinel(tokenizer, index):
    """At the exhausted object slot the real (duplicate) object is blocked and the
    sentinel object is emitted instead, then the state returns to free generation."""
    T = _triple_ids(tokenizer)
    V = len(tokenizer)
    obj = _object_slot(tokenizer, T)
    proc = ConstrainedLogitsProcessor(index, None, tokenizer=tokenizer, sentinel=True)
    sent_ids = proc._sentinel_ids()

    st = _state(tokenizer)
    allowed = _allowed(proc, st, T[:obj], V)
    assert T[obj] not in allowed, "duplicate real object must be blocked"
    assert sent_ids[0] in allowed, "sentinel object must be offered instead"
    assert st.sentinel_remaining, "sentinel emission should be in progress"

    # Drive the rest of the sentinel emission token by token.
    emitted = [sent_ids[0]]
    while st.sentinel_remaining:
        mask = torch.full((1, V), float("-inf"))
        proc.constrained_generation([], mask, 0, st)
        emitted.append((mask[0] == 0).nonzero(as_tuple=True)[0].item())

    assert emitted == sent_ids
    assert tokenizer.decode(emitted) == proc.sentinel_text
    assert st.state == 0, "state should return to normal (free) generation after the sentinel"


def test_sentinel_off_by_default(tokenizer, index):
    proc = ConstrainedLogitsProcessor(index, None, tokenizer=tokenizer)
    assert proc.sentinel is False
