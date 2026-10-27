"""Answer generation used in all evaluations (greedy decoding, 128 new tokens)."""
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

PROMPT_TEMPLATE = "\nAnswer question:\n{question}\n"


class StopOnCloseBracket(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_token_id = tokenizer.convert_tokens_to_ids("]")

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.stop_token_id


@torch.no_grad()
def generate(model, tokenizer, question, max_new_tokens=128, temperature=None):
    """Answers `question`; interventions apply if a hook context (valueheads.hooks) is active.

    Greedy by default; with `temperature`, samples with top-p 0.9 (robustness runs, Appendix I).
    """
    inputs = tokenizer(PROMPT_TEMPLATE.format(question=question), return_tensors="pt").to(model.device)
    sampling = dict(do_sample=True, temperature=temperature, top_p=0.9) if temperature else dict(do_sample=False)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        **sampling,
        stopping_criteria=StoppingCriteriaList([StopOnCloseBracket(tokenizer)]),
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
