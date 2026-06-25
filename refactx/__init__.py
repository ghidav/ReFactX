from refactx.index import load_index, populate_postgres_index
from refactx.prompt_base import PROMPT_TEMPLATE

def apply_prompt_template(tokenizer, prompt_template=PROMPT_TEMPLATE, question=None, **chat_template_kwargs):
    # Extra kwargs are forwarded to ``apply_chat_template`` (e.g.
    # ``enable_thinking=False`` for Qwen3-style thinking models, whose <think>
    # ramble otherwise breaks the ReFactX `Fact:` protocol). Templates that do
    # not use a given kwarg simply ignore it.
    if question is None:
        # only prompt for caching
        return tokenizer.apply_chat_template(prompt_template, tokenize=False, add_generation_prompt=False, **chat_template_kwargs)
    else:
        question_w_role = {'role':'user', 'content': question}
        return tokenizer.apply_chat_template(prompt_template + [question_w_role], tokenize=False, add_generation_prompt=True, **chat_template_kwargs)

def get_constrained_logits_processor(tokenizer, index, num_beams=1, num_batches=1, return_list=False):
    from refactx.generate import get_constrained_logits_processor as _base
    return _base(tokenizer, index, num_beams=num_beams, num_batches=num_batches, return_list=return_list)

def patch_model(model):
    from refactx.generate import patch_model as _base
    return _base(model)

def get_constrained_states():
    from refactx.generate import CONSTRAINED_STATES
    return CONSTRAINED_STATES

def _read_version_from_pyproject():
	try:
		import toml
		from pathlib import Path
		p = Path(__file__).resolve().parents[1] / 'pyproject.toml'
		data = toml.load(p)
		# PEP 621 field
		version = data.get('project', {}).get('version')
		if version:
			return version
	except Exception:
		pass
	return '0.0.0'

__version__ = _read_version_from_pyproject()

__all__ = [load_index, populate_postgres_index, apply_prompt_template, get_constrained_logits_processor, patch_model, get_constrained_states]