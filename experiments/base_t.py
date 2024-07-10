import json
import typing
from typing import List
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import time
import unicodedata
import os
import shutil

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
# 数据集类
class MQuAKE_T(Dataset):
    """
    Dataset class for loading MQuAKE-T data.
    """
    def __init__(self, data_dir: str, size: typing.Optional[int] = None, *args, **kwargs):
        data_dir = Path(data_dir)
        mquake_loc = data_dir / "MQuAKE-T.json"
        if not mquake_loc.exists():
            remote_url = f"{REMOTE_ROOT}/MQuAKE-CF-3k.json"
            print(f"{mquake_loc} does not exist. Downloading from {remote_url}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(remote_url, mquake_loc)
        
        with open(mquake_loc, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]
        
        print(f"Loaded MQuAKE-CF-3k dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

# Helper functions
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def compute_rewrite_quality_mquake(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict,
    multi_hop_prompt: str,
    rel_prompts: dict
) -> typing.Dict:
    multi_hop_accuracy, generated_answers, edit_success_rate, instance_accuracy = calculate_multi_hop_accuracy(
        model, tokenizer, record, multi_hop_prompt, rel_prompts
    )

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

def calculate_multi_hop_accuracy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict,
    multi_hop_prompt: str,
    rel_prompts: dict
):
    correct_responses = 0
    generated_answers = []
    questions = record['questions']
    correct_answer = record['answer']
    answer_aliases = record.get('answer_alias', [])
    extended_answers = record.get('answer_extended', [])
    requested_rewrite = record['requested_rewrite']
    single_hops = record['single_hops']
    
    all_questions = questions

    for question in all_questions:
        full_prompt = multi_hop_prompt + "\nQ: " + question 
        clear_torch_cache()
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
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(multi_hop_prompt, "").strip()
        if "A:" in generated_text:
            generated_answer = generated_text.split("A:")[1].strip().split("Q:")[0].strip()
        else:
            generated_answer = generated_text.strip()
        
        print("Question:", question)
        print("Generated Answer:\n", generated_answer)
        
        if question in questions:
            generated_answers.append(generated_answer)
            if (correct_answer.lower() in generated_answer.lower() or
                    any(alias.lower() in generated_answer.lower() for alias in answer_aliases) or
                    any(ext_answer.lower() in generated_answer.lower() for ext_answer in extended_answers)):
                correct_responses += 1

    multi_hop_accuracy = correct_responses / len(questions)

    # Editwise accuracy
    edit_success_count = 0
    
    # We assume that all single_hops have the same relation_id in one case
    if requested_rewrite:
        relation_id = requested_rewrite[0]['relation_id']
    else:
        relation_id = ""

    rel_prompt = rel_prompts.get(relation_id, "")
    
    for single_hop in single_hops:
        question = single_hop['question']
        full_prompt = rel_prompt + "\nQ: " + question
        clear_torch_cache()
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
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        
        print("Single Hop Question:", question)
        print("Generated Answer:\n", generated_answer)
        
        if (single_hop['answer'].lower() in generated_answer.lower() or
                any(alias.lower() in generated_answer.lower() for alias in single_hop['answer_alias'])):
            edit_success_count += 1
    
    editwise_accuracy = edit_success_count / len(single_hops)
    instance_accuracy = 1 if edit_success_count == len(single_hops) else 0

    return multi_hop_accuracy, generated_answers, editwise_accuracy, instance_accuracy






def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
):
    """
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """

    # Unroll prompts and tokenize
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    batch_size = input_ids.size(0)

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits, past_key_values = model_out.logits, model_out.past_key_values
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for x in txt
    ]

    return txt

def main(
    model_name: str,
    ds_name: str,
    dataset_size_limit: int,
    generation_test_interval: int,
    dir_name: str,
    multi_hop_prompt: str,
    rel_prompts: dict
):
    current_dir = Path.cwd()
    results_dir = current_dir / "results" / dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {results_dir}")

    print("Instantiating model")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token

    print("Loading dataset")
    data_dir = Path("data")
    dataset = MQuAKE_T(data_dir, size=dataset_size_limit)

    new_results_dir = results_dir / "evaluation" / "original_model"
    new_results_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluating all data with the original model...")
    for record_chunks in chunks(dataset, 1):
        for record in record_chunks:
            out_file = Path(new_results_dir / f"original_model_case_{record['case_id']}.json")
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue
            
            start = time()
            exec_time = time() - start
            metrics = {
                "case_id": record["case_id"],
                "grouped_case_ids": [record["case_id"]],
                "num_edits": 0,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": compute_rewrite_quality_mquake(
                    model,
                    tok,
                    record,
                    multi_hop_prompt,
                    rel_prompts
                ),
            }

            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)
            print(f"Evaluation took {time() - start} seconds")

if __name__ == "__main__":
    with open('multihop-prompts.txt', 'r') as f:
        multi_hop_prompt = f.read()
    with open('rel-prompts.json', 'r') as f:
        rel_prompts = json.load(f) 
    main(
        model_name="EleutherAI/gpt-j-6B",
        ds_name="mquake",
        dataset_size_limit=3000,
        generation_test_interval=1,
        dir_name="mquake_t",
        multi_hop_prompt=multi_hop_prompt,
        rel_prompts=rel_prompts,
    )
