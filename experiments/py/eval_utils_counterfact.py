import typing
import nltk
import numpy as np
import scipy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain
from dsets import AttributeSnippets
from util.generate import generate_fast
from util.perplexity import perplexity
import re
import json
def compute_rewrite_quality_counterfact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???

    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]
    generation_prompts = record["generation_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
    ]
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(paraphrase_prompts))],
        [1 for _ in range(len(neighborhood_prompts))],
    ]
    # Flatten all the evaluated prefixes into one list.
    probs, targets_correct = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new["str"],
        target_true["str"],
    )
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    } | {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    }

    if snips is not None:
        # Gather reference texts
        rel_id = record["requested_rewrite"]["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        essence_texts = [
            x["text"]
            for x in snips[rel_id][target_new["id"]]
            if x["name"] == record["requested_rewrite"]["subject"]
        ]
        assert (
            len(consistency_texts) > 0
        ), "Must have consistency texts to evaluate generation"
        gen_stats = test_generation(
            model,
            tok,
            generation_prompts,
            consistency_texts,
            essence_texts,
            vec,
        )
        ret.update(gen_stats)

    return ret


import torch
import typing
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import shutil
import os

def clear_torch_cache():
    cache_dir = os.path.expanduser('~/.cache/torch/kernels')
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            try:
                if os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"Failed to remove {item_path}. Reason: {e}")

def compute_rewrite_quality_mquake(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict,
    snips: typing.Optional[AttributeSnippets] = None,
    vec: typing.Optional[TfidfVectorizer] = None
) -> typing.Dict:
    """
    Evaluates the rewritten model on a MQuAKE dataset record for multiple metrics including
    edit-wise success rate, instance-wise accuracy, and multi-hop accuracy.

    :param model: The language model.
    :param tokenizer: The tokenizer.
    :param record: A single record from the MQuAKE dataset.
    :param snips: Optional, attribute snippets for reference texts.
    :param vec: Optional, a TF-IDF vectorizer.
    :return: A dictionary with evaluation metrics.
    """
    multi_hop_accuracy, edit_success_rate, instance_accuracy, generated_answers = calculate_metrics(
        model, tokenizer, record
    )

    # 打印各个指标
    print(f"Multi-hop Accuracy: {multi_hop_accuracy}")
    print(f"Edit-wise Success Rate: {edit_success_rate}")
    print(f"Instance-wise Accuracy: {instance_accuracy}")

    return {
        'multi_hop_accuracy': multi_hop_accuracy,
        'edit_success_rate': edit_success_rate,
        'instance_accuracy': instance_accuracy,
        'questions': record['questions'],
        'original_answers': [rw['target_true']['str'] for rw in record['requested_rewrite']],
        'generated_answers': generated_answers,
    }

def calculate_metrics(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict
):
    """
    Calculate multi-hop accuracy, edit-wise success rate, and instance-wise accuracy.

    :param model: The language model.
    :param tokenizer: The tokenizer.
    :param record: A single record from the MQuAKE dataset.
    :return: Multi-hop accuracy, edit-wise success rate, instance-wise accuracy, and generated answers.
    """
    # Load multi-hop prompt
    with open('multihop-prompts.txt', 'r') as f:
        multi_hop_prompt = f.read()

    # Load relation prompts
    with open('rel-prompts.json', 'r') as f:
        rel_prompts = json.load(f)

    correct_responses = 0
    success_count = 0
    generated_answers = []

    questions = record['questions']
    correct_answer = record['new_answer']
    answer_aliases = record.get('new_answer_alias', [])
    new_single_hops = record['new_single_hops']

    # Get the relation_id from the requested_rewrite
    relation_id = record['requested_rewrite'][0]['relation_id']

    # Calculate multi-hop accuracy
    for question in questions:
        # Clear cache
        #clear_torch_cache()

        # Generate answer using model.generate()
        full_prompt = multi_hop_prompt + "\nQ: " + question
        inputs = tokenizer(full_prompt, return_tensors='pt').to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            top_k=5,
            do_sample=True
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = generated_text.replace(multi_hop_prompt, "").strip()
        if "A:" in generated_answer:
            generated_answer = generated_answer.split("A:")[1].strip().split("Q:")[0].strip()
        else:
            generated_answer = generated_answer.strip()

        # Append generated answer
        generated_answers.append(generated_answer)

        # Debugging information
        print(f"Question: {question}")
        print(f"Generated Answer: {generated_answer}")

        if correct_answer.lower() in generated_answer.lower() or any(alias.lower() in generated_answer.lower() for alias in answer_aliases):
            correct_responses += 1

    # Calculate edit-wise success rate and instance-wise accuracy
    for single_hop in new_single_hops:
        cloze_question = single_hop['cloze']
        full_prompt = cloze_question
        #clear_torch_cache()
        inputs = tokenizer(full_prompt, return_tensors='pt').to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            top_k=5,
            do_sample=True
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        print("Single Hop Question:", cloze_question)
        print("Generated Answer:\n", generated_text)

        if single_hop['answer'].lower() in generated_text.lower() or any(alias.lower() in generated_text.lower() for alias in single_hop.get('answer_alias', [])):
            success_count += 1

    multi_hop_accuracy = correct_responses / len(questions)
    editwise_accuracy = success_count / len(new_single_hops)
    instance_accuracy = 1 if success_count == len(new_single_hops) else 0

    return multi_hop_accuracy, editwise_accuracy, instance_accuracy, generated_answers

'''import unicodedata
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

def calculate_metrics(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict
):
    """
    Calculate multi-hop accuracy, edit-wise success rate, and instance-wise accuracy.

    :param model: The language model.
    :param tokenizer: The tokenizer.
    :param record: A single record from the MQuAKE dataset.
    :return: Multi-hop accuracy, edit-wise success rate, instance-wise accuracy, and generated answers.
    """
    correct_responses = 0
    success_count = 0
    all_facts_recalled = True
    generated_answers = []

    questions = record['questions']
    correct_answer = record['new_answer']
    answer_aliases = record.get('new_answer_alias', [])
    requested_rewrite = record['requested_rewrite']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 获取模型所在的设备
    model.to(device)  # 确保模型在 GPU 上
    tokenizer.pad_token_id = tokenizer.eos_token_id  # 设置 pad_token_id

    for question in questions + [rw['prompt'].format(rw['subject']) for rw in requested_rewrite]:
        # 使用 model.generate 函数生成答案
        inputs = tokenizer([question], padding=True, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # 生成时调试信息
        print(f"input_ids device: {input_ids.device}, attention_mask device: {attention_mask.device}")
        clear_torch_cache
        # Generate outputs using model.generate
        generated_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=100,
            do_sample=True,
            top_k=5,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode the generated outputs
        generated_texts = [
            tokenizer.decode(output.cpu(), skip_special_tokens=True).split(tokenizer.eos_token)[0].strip()
            for output in generated_outputs
        ]

        # Normalize the text
        generated_texts = [
            unicodedata.normalize("NFKD", text).replace("\n\n", " ").replace("\n", " ").replace("", "")
            for text in generated_texts
        ]

        generated_answer = generated_texts[0]

        # 获取生成文本的回答部分，针对 multi-hop accuracy
        if question in questions:
            generated_answers.append(generated_answer)

            # Debugging information
            print(f"Question: {question}")
            print(f"Generated Answer: {generated_answer}")

            if correct_answer.lower() in generated_answer.lower() or any(alias.lower() in generated_answer.lower() for alias in answer_aliases):
                correct_responses += 1

        # 针对 edit-wise success rate 和 instance-wise accuracy
        if question not in questions:
            matching_rewrites = [rw for rw in requested_rewrite if rw['prompt'].format(rw['subject']) == question]

            if not matching_rewrites:
                print(f"No matching rewrite found for question: {question}")
                continue

            target_new = matching_rewrites[0]['target_new']['str']

            if target_new.lower() in generated_answer.lower():
                success_count += 1

    # Check if all facts are recalled
    all_facts_recalled = (success_count == len(requested_rewrite))

    multi_hop_accuracy = correct_responses / len(questions)
    edit_success_rate = success_count / len(requested_rewrite)
    instance_accuracy = 1 if all_facts_recalled else 0

    return multi_hop_accuracy, edit_success_rate, instance_accuracy, generated_answers'''














def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct


def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }

    if len(essence_texts) > 0:
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"essence_score": ppl, "essence_text": essence_texts})

    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()

# 在这里添加你的评估代码
