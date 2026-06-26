from transformers import LogitsProcessorList
from refactx.index import DictIndex
from transformers.generation.logits_process import LogitsProcessor
import torch
from copy import deepcopy
import math
import types

from refactx.index import EmptyIndexException, TripleNotFoundException


def patch_model(model, verbose=True):
    _get_running_beams_for_next_iteration_original = model.__class__._get_running_beams_for_next_iteration
    
    def _get_running_beams_for_next_iteration_patch(self,*args, **kwargs):
        global CONSTRAINED_STATES
        running_sequences, running_beam_scores, running_beam_indices = _get_running_beams_for_next_iteration_original(self, *args, **kwargs)
        CONSTRAINED_STATES.beam_idx = running_beam_indices
        return running_sequences, running_beam_scores, running_beam_indices    
    
    model._get_running_beams_for_next_iteration = types.MethodType(_get_running_beams_for_next_iteration_patch, model)

    if verbose:
        print('WARNING: this patching method relies on shared mutable global state to support constrained generation with beam search. It is not thread-safe and may produce incorrect results in concurrent or multi-process setups (e.g. multiple workers).')

# def get_constrained_logits_processor(tokenizer, index, num_beams=1, num_batches=1, return_list=True):
def get_constrained_logits_processor(tokenizer, index, num_beams, num_batches, return_list):
    states = [[PatternConstrainedState(
                    pattern = 'Fact:',
                    tokenizer = tokenizer,
                    cache_index = DictIndex(),
                    subtree_cache = DictIndex(),
                )]]

    CONSTRAINED_STATES.__init__(states,
                num_beams=num_beams,
                num_batches =num_batches,
        )

    constrained_processor = ConstrainedLogitsProcessor(
        index=index,
        states=CONSTRAINED_STATES, tokenizer=tokenizer)

    if return_list:
        logits_processor_list = LogitsProcessorList([
            constrained_processor
        ])
        return logits_processor_list
    else:
        return constrained_processor

class ConstrainedStateList():
    # states is list of list [num_batches, num_beams]
    def __init__(self, states, num_beams = 1, num_batches = 1, debug=False, debug_tokenizer=None):
        self.states = states
        if states != []:
            assert isinstance(states, list) and isinstance(states[0], list), 'ERROR: states is not a list of lists'
            assert len(states) == num_batches and len(states[0]) == num_beams, 'ERROR: states size does not match num_batches or num_beams'
        self.num_beams = num_beams
        self.num_batches = num_batches # used for computing beam id
        self.beam_idx = [] # torch.tensor([-1]*num_batches*num_beams).view(num_batches,num_beams,1) # running beam idx

        self.debug = debug
        self.debug_tokenizer = debug_tokenizer
        assert not self.debug or self.debug_tokenizer is not None

        self.num_permutations = 0 # for debugging

    def __getitem__(self, key):
        if isinstance(key, tuple):
            # TODO test # seems it is not working as expected
            batch_key, beam_key = key

            if isinstance(batch_key, int) and isinstance(beam_key, int):
                return self.states[batch_key][beam_key]
            else:
                # Convert int to slice to normalize
                if isinstance(batch_key, int):
                    batch_key = slice(batch_key, batch_key + 1)
                if isinstance(beam_key, int):
                    beam_key = slice(beam_key, beam_key + 1)

                # Slice the states
                sliced_states = [row[beam_key] for row in self.states[batch_key]]

                new_num_batches = len(sliced_states)
                new_num_beams = len(sliced_states[0]) if sliced_states else 0

                return ConstrainedStateList(
                    sliced_states,
                    num_beams=new_num_beams,
                    num_batches=new_num_batches
                )

        elif isinstance(key, slice):
            return ConstrainedStateList(
                self.states[key],
                num_beams=self.num_beams,
                num_batches=len(self.states[key])
            )
        elif isinstance(key, int):
            return self.states[key]
        else:
            raise TypeError(f"Invalid argument type: {type(key)}")

    def __len__(self):
        return len(self.states) * len(self.states[0])

    def reset(self):
        for batch in self.states:
            for state in batch:
                state.reset()

    def get_batch_idx(self, idx):
        return int(idx // self.num_beams)
    
    def get_beam_idx(self, idx):
        return int(idx % self.num_beams)
    
    def get_last_beam_z(self):
        z = -1
        ids = (self.beam_idx[0, 0] != -1).nonzero(as_tuple=False).squeeze()
        if ids.numel() > 0:
            if ids.dim() == 0:
                z = ids.item()
            else:
                z = ids[-1].item()
        return z

    def beam_permutation(self):
        if len(self.beam_idx) > 0: # ignore first call
            assert self.beam_idx.shape[0] * self.beam_idx.shape[1] == self.num_beams * self.num_batches, f'ERROR: beam_idx size unexpected: {len(self.beam_idx)} != {self.num_beams} * {self.num_batches}'
            # copies = self[:,:] # new object
            copies = []
            for batch_i in range(self.num_batches):
                batch_copies = []
                for beam_i in range(self.num_beams):
                    batch_copies.append(self[batch_i, beam_i].dump())
                copies.append(batch_copies)
            # copies = [[self[batch_i, beam_i].dump() for beam_i in range(self.num_beams)] for batch_i in range(self.num_batches)]
            last_beam_z = self.get_last_beam_z()
            # skip first call
            if last_beam_z >= 0:
                if self.debug:
                    for batch_idx in range(self.beam_idx.shape[0]):
                        for num_beam in range(self.beam_idx.shape[1]):
                            print((batch_idx, num_beam), end='')
                            self.states[batch_idx][num_beam].print_debug(self.debug_tokenizer)

                for batch_idx in range(self.beam_idx.shape[0]):
                    for num_beam in range(self.beam_idx.shape[1]):
                        replacement_idx = self.beam_idx[batch_idx, num_beam, last_beam_z]
                        replacement_batch_idx = self.get_batch_idx(replacement_idx)
                        local_beam_idx = self.get_beam_idx(replacement_idx)
                        assert replacement_batch_idx == batch_idx, f'ERROR: permutating between different batches! {replacement_batch_idx} --> {batch_idx}, with num_beams {self.num_beams}. replacement_idx {replacement_idx}'
                        # copy only when to change
                        if num_beam != local_beam_idx:
                            # self.states[batch_idx][num_beam].copy(copies[batch_idx][local_beam_idx], copy=True)
                            self.states[batch_idx][num_beam].load(copies[batch_idx][local_beam_idx], copy=True)
                            self.num_permutations += 1
                            if self.debug:
                                print(f'permutation {self.num_permutations}: ({batch_idx},{local_beam_idx}) into {batch_idx}{num_beam}')

"""
Pattern should be recognized as soon as it is generated. Usually you want to end it with $
"""
class PatternConstrainedState():
    def __init__(self, pattern, tokenizer, cache_index, subtree_cache, state=0, debug=False, regex_window=10) -> None:

        self.NORMAL_GENERATION = 0 # even numbers for normal
        self.CONSTRAINED_GENERATION = 1 # odd numbers for constrained

        self.token_ids = [] # keep all the token ids

        self.tokenizer = tokenizer
        self.regex_window = regex_window # regex will be performed on the last N tokens

        # if the switch pattern is finally found --> CONSTRAINED_GENERATION
        self.pattern = pattern

        self.state = state

        self.cursor = 0 # how many tokens since last change in state

        self.cache_index = cache_index
        self.generated_triples = []

        self.subtree_cache = subtree_cache

        # Decode-time sentinel: token ids still to be emitted for a
        # "no further records" object once a subject-relation is exhausted.
        self.sentinel_remaining = []

        self._first_call = True

        self.debug = debug
        self.debug_history = []

    def print_debug(self, tokenizer, print_class=False, end_with_newline=True):
        if print_class:
            print('{} '.format(self), end='')
        # for item in self.debug_history:
        #     print('{} ({}) --> {}, '.format(tokenizer.decode(item['token']), item['token'], item['state']), end='')
        if end_with_newline:
            print()

    def first_call(self):
        if self._first_call:
            self._first_call = False
            return True
        else:
            return False

    def cache_add(self, sequence):
        self.cache_index.add(sequence, new_leaf=True)
        # removing end of triple
        self.generated_triples.append(sequence[:-1])

    def is_constrained(self):
        return self.state % 2 == self.CONSTRAINED_GENERATION

    def end_of_triple_reset(self):
        self.subtree_cache.reset()
        self.sentinel_remaining = []
        # reset to normal generation
        self.state = 0

    def reset(self):
        self.state = 0
        self.token_ids = []
        self.history = ()
        self.cursor = 0
        self.generated_triples = []
        self.cache_index.reset()
        self.end_of_triple_reset()

    def copy(self, other, copy=True):
        self.pattern = other.pattern
        self.state = other.state
        self.tokenizer = other.tokenizer
        self.regex_window = other.regex_window
        self.token_ids = deepcopy(other.token_ids)

        self.history = other.history  # Assuming it's immutable or should be shallow copied
        self.cursor = other.cursor

        self.generated_triples = other.generated_triples.copy() if copy else other.generated_triples
        self.cache_index = deepcopy(other.cache_index) if copy else other.cache_index
        self.subtree_cache = deepcopy(other.subtree_cache) if copy else other.subtree_cache

        self.debug = other.debug
        self.debug_history = deepcopy(other.debug_history) if copy else other.debug_history


    def update(self, new_token):
        state = self.state
        self.token_ids.append(new_token)
        self.cursor += 1

        text = self.tokenizer.decode(self.token_ids[-self.regex_window:])

        _match = text.endswith(self.pattern)
        if _match:
            state = self.CONSTRAINED_GENERATION

        self._update_state(state)

        if self.debug:
            self.debug_history.append({
                'state': self.state,
                'token': new_token,
                'token_ids': deepcopy(self.token_ids)
            })

    def _update_state(self, state, initial_cursor = 0):
        if state != self.state:
            self.history = (self.state, self.cursor)

            self.state = state
            self.cursor = initial_cursor
            self.token_ids = []

    def get_cursor(self):
        return self.cursor

class ConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, index, states, tokenizer=None, error_strategy=0, avoid_duplicates=True,
                 sentinel=False, sentinel_text='no further records>'):
        self.index = index
        self.states = states
        self.error_strategy = error_strategy
        self.avoid_duplicates = avoid_duplicates

        # When ``sentinel`` is on, an exhausted subject-relation is not pruned
        # from the trie (which would force the model onto a different relation).
        # Instead the relation stays selectable and its object slot yields a
        # fixed "no further records" object, so the model gets an explicit
        # "nothing more here" signal. ``sentinel_text`` is the object value
        # (without the leading ``<``, which the model emits as the object opener).
        self.sentinel = sentinel
        self.sentinel_text = sentinel_text
        self._sentinel_ids_cache = None

        self.ERROR_STRATEGY_WARN = 0
        self.ERROR_STRATEGY_FAIL = 1

        self.tokenizer=tokenizer # for debugging

    def _sentinel_ids(self):
        if self._sentinel_ids_cache is None:
            self._sentinel_ids_cache = self.tokenizer.encode(
                self.sentinel_text, add_special_tokens=False)
        return self._sentinel_ids_cache

    def _begin_sentinel(self, state, mask, mask_idx):
        ids = self._sentinel_ids()
        mask[mask_idx, :] = -math.inf
        mask[mask_idx, ids[0]] = 0
        state.sentinel_remaining = list(ids[1:])
        if not state.sentinel_remaining:
            state.end_of_triple_reset()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        assert input_ids.shape[0] == len(self.states), \
            f'Error: number of states ({len(self.states)}) should match `num_batches * num_beams` ({input_ids.shape[0]})'

        self.states.beam_permutation()

        # TODO create a mask of zeros same shape as scores and same device
        mask = torch.zeros_like(scores)

        for i in range(input_ids.shape[0]):
            batch_idx = self.states.get_batch_idx(i)
            beam_i = self.states.get_beam_idx(i)
            sequence = input_ids[i].tolist()

            if not self.states[batch_idx, beam_i].first_call():
                last_token = sequence[-1]
                self.states[batch_idx, beam_i].update(last_token)

            if self.states[batch_idx, beam_i].is_constrained(): # odd number means constrained generation
                # constrained generation
                mask[i] = -math.inf # set for all tokens by default
                constrain_generation_sequence = sequence[len(sequence) - self.states[batch_idx, beam_i].get_cursor():]
                self.constrained_generation(
                    constrain_generation_sequence, mask, i, state=self.states[batch_idx, beam_i])

                # else:
                #     # normal generation
                #     # scores are not altered
                #     pass

        scores_processed = scores + mask

        return scores_processed

    def constrained_generation(self, sequence, mask: torch.FloatTensor, mask_idx, state):
        # Mid-sentinel: keep emitting the "no further records" object token by token.
        if state.sentinel_remaining:
            nxt = state.sentinel_remaining.pop(0)
            mask[mask_idx, :] = -math.inf
            mask[mask_idx, nxt] = 0
            if not state.sentinel_remaining:
                state.end_of_triple_reset()
            return

        possible_tokens, _ = self.index.next_tokens(sequence, state = state)

        if not self.sentinel:
            # ---- original behaviour: prune exhausted branches ----
            if self.avoid_duplicates:
                try:
                    visited_tokens, _ = state.cache_index.next_tokens(sequence)
                    state.cache_index.subtract_tokens(possible_tokens, visited_tokens)
                except EmptyIndexException:
                    pass
                except TripleNotFoundException:
                    pass

            keys = list(possible_tokens.keys())
            if len(keys) == 0:
                state.cache_add(sequence)
                state.end_of_triple_reset()
                mask[mask_idx, :] = 0
            else:
                mask[mask_idx, keys] = 0
            return

        # ---- sentinel-enabled behaviour ----
        # Split continuations into `live` (new leaves remain) and `exhausted`
        # (every leaf already generated) instead of deleting the exhausted ones.
        live = dict(possible_tokens)
        exhausted = {}
        if self.avoid_duplicates:
            try:
                visited_tokens, _ = state.cache_index.next_tokens(sequence)
                for tok, total in list(possible_tokens.items()):
                    if total - visited_tokens.get(tok, 0) <= 0:
                        exhausted[tok] = total
                        del live[tok]
            except EmptyIndexException:
                pass
            except TripleNotFoundException:
                pass

        if len(possible_tokens) == 0:
            # Genuine end of a real triple in the trie.
            state.cache_add(sequence)
            state.end_of_triple_reset()
            mask[mask_idx, :] = 0
            return

        # Object slot starts after the second "> <" of "<S> <R> <O>".
        in_object = (self.tokenizer or state.tokenizer).decode(sequence).count('> <') >= 2
        live_keys = list(live.keys())

        if in_object:
            if live_keys:
                # New object value(s) still available: emit them normally.
                mask[mask_idx, live_keys] = 0
            else:
                # Object fully visited -> emit the sentinel instead of forcing a
                # duplicate (object exhausted) or rerouting.
                self._begin_sentinel(state, mask, mask_idx)
        else:
            # Subject / relation: keep exhausted branches selectable so the model
            # is not forced off the relation it wants.
            mask[mask_idx, live_keys + list(exhausted.keys())] = 0

CONSTRAINED_STATES = ConstrainedStateList([])
