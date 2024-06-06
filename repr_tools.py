from copy import deepcopy
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook


def get_reprs_at_word_tokens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """
    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)
    print("Computed indices:", idxs)
    return get_reprs_at_idxs(
        model,
        tok,
        [context_templates[i].format(words[i]) for i in range(len(words))],
        idxs,
        layer,
        module_template,
        track,
    )


def get_words_idxs_in_templates(
    tok: AutoTokenizer, context_templates: List[str], words: List[str], subtoken: str
) -> List[List[int]]:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "Each template should only contain one fill-in placeholder."

    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    prefixes, suffixes = [
        tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    ], [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]
    words = deepcopy(words)

    # Pre-process tokens
    for i, prefix in enumerate(prefixes):
        if len(prefix) > 0 and prefix[-1] == " ":
            prefix = prefix[:-1]
            prefixes[i] = prefix
            words[i] = f" {words[i].strip()}"

    # Tokenize to determine lengths
    batch_tok = tok([*prefixes, *words, *suffixes], add_special_tokens=False, is_split_into_words=False)
    prefixes_len, words_len, suffixes_len = [
        [len(tok.convert_ids_to_tokens(batch_tok['input_ids'][i])) for i in range(len(prefixes))],
        [len(tok.convert_ids_to_tokens(batch_tok['input_ids'][i + len(prefixes)])) for i in range(len(words))],
        [len(tok.convert_ids_to_tokens(batch_tok['input_ids'][i + len(prefixes) + len(words)])) for i in range(len(suffixes))]
    ]

    # Compute indices of last tokens
    if subtoken == "last":
        return [[prefixes_len[i] + words_len[i] - 1] for i in range(len(words))]
    elif subtoken == "first":
        return [[prefixes_len[i]] for i in range(len(words))]
    elif subtoken == "first_after_last":
        return [[prefixes_len[i] + words_len[i]] for i in range(len(words))]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")


def get_reprs_at_idxs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    contexts: List[str],
    idxs: List[List[int]],
    layer: int,
    module_template: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """
    def _batch(n):
        for i in range(0, len(contexts), n):
            yield contexts[i : i + n], idxs[i : i + n]

    assert track in {"in", "out", "both"}
    both = track == "both"
    tin, tout = (track == "in" or both), (track == "out" or both)
    module_name = module_template.format(layer)
    to_return = {"in": [], "out": []}

    def _process(cur_repr, batch_idxs, key):
        nonlocal to_return
        # Handle tensor directly if it's not a tuple
        if not isinstance(cur_repr, tuple):
            if not isinstance(cur_repr, torch.Tensor):
                print("Unexpected data type:", type(cur_repr))
                return
        else:
            if len(cur_repr) > 0:
                cur_repr = cur_repr[0]
            else:
                print("Empty tuple received")
                return
        
        for i, idx_list in enumerate(batch_idxs):
            if idx_list[0] < len(cur_repr[i]):
                to_return[key].append(cur_repr[i][idx_list].mean(0))
            else:
                print(f"Index {idx_list} out of range for tensor index {i}.")

    for batch_contexts, batch_idxs in _batch(n=128):
        contexts_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=tin,
                retain_output=tout,
            ) as tr:
                model(**contexts_tok)

        if tin:
            _process(tr.input, batch_idxs, "in")
        if tout:
            _process(tr.output, batch_idxs, "out")

    # Ensure to_return has values even if empty
    in_tensor = torch.stack(to_return["in"], 0) if len(to_return["in"]) > 0 else torch.tensor([])
    out_tensor = torch.stack(to_return["out"], 0) if len(to_return["out"]) > 0 else torch.tensor([])

    if len(to_return) == 1:
        return in_tensor if tin else out_tensor
    else:
        return in_tensor, out_tensor
