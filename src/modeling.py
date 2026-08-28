"""Model-independent serialization, prompting, and reproducibility helpers."""

import json
import random

import numpy as np


def llama_v2_prompt(messages, system_prompt=""):
    """Format messages with the Llama-2-chat template used by TalkTuner."""
    b_inst, e_inst = "[INST]", "[/INST]"
    b_sys, e_sys = "<<SYS>>\n", "\n<</SYS>>\n\n"
    bos, eos = "<s>", "</s>"
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages
    messages = [{
        "role": messages[1]["role"],
        "content": b_sys + messages[0]["content"] + e_sys + messages[1]["content"],
    }] + messages[2:]
    parts = [
        f"{bos}{b_inst} {prompt['content'].strip()} {e_inst} "
        f"{answer['content'].strip()} {eos}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    if messages[-1]["role"] == "user":
        parts.append(f"{bos}{b_inst} {messages[-1]['content'].strip()} {e_inst}")
    return "".join(parts)


def set_seed(seed, use_torch=False):
    random.seed(seed)
    np.random.seed(seed)
    if use_torch:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def option_token_ids(tokenizer, max_options=12):
    import string

    ids_by_letter = {}
    for letter in string.ascii_uppercase[:max_options]:
        # The leading space matches how answer letters occur after ``Answer:``.
        ids = tokenizer(f" {letter}", add_special_tokens=False).input_ids
        if not ids:
            raise ValueError(f"Tokenizer produced no token for option {letter}.")
        ids_by_letter[letter] = ids[-1]
    return ids_by_letter


def serialize_distribution(values):
    # Fixed precision keeps result files compact and stable across runs.
    return json.dumps([round(float(value), 8) for value in values], ensure_ascii=False)


def json_array(values):
    rounded = [round(value, 8) if isinstance(value, float) else value for value in values]
    return json.dumps(rounded, ensure_ascii=False)
