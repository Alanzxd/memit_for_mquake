import json
import typing
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import time

# 数据集类
class MQuAKE_T(Dataset):
    """
    Dataset class for loading MQuAKE-T data.
    """
    def __init__(self, data_dir: str, size: typing.Optional[int] = None, *args, **kwargs):
        data_dir = Path(data_dir)
        mquake_loc = data_dir / "MQuAKE-T.json"
        if not mquake_loc.exists():
            remote_url = f"{REMOTE_ROOT}/MQuAKE-T.json"
            print(f"{mquake_loc} does not exist. Downloading from {remote_url}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(remote_url, mquake_loc)
        
        with open(mquake_loc, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]
        
        print(f"Loaded MQuAKE-T dataset with {len(self)} elements")

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
    record: dict
) -> typing.Dict:
    multi_hop_accuracy, edit_success_rate, instance_accuracy, generated_answers = calculate_metrics(
        model, tokenizer, record
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

def calculate_metrics(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict
):
    correct_responses = 0
    success_count = 0
    all_facts_recalled = True
    generated_answers = []

    questions = record['questions']
    correct_answer = record['new_answer']
    answer_aliases = record.get('new_answer_alias', [])
    requested_rewrite = record['requested_rewrite']

    for question in questions + [rw['prompt'].format(rw['subject']) for rw in requested_rewrite]:
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=100,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,  # Ensure sampling is used
            top_k=5
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        generated_answers.append(generated_text)

        if question in questions:
            print(f"Question: {question}")
            print(f"Generated Answer: {generated_text}")

            if correct_answer.lower() in generated_text.lower() or any(alias.lower() in generated_text.lower() for alias in answer_aliases):
                correct_responses += 1

        if question not in questions:
            matching_rewrites = [rw for rw in requested_rewrite if rw['prompt'].format(rw['subject']) == question]
            
            if not matching_rewrites:
                print(f"No matching rewrite found for question: {question}")
                continue
            
            target_new = matching_rewrites[0]['target_new']['str']

            if target_new.lower() in generated_text.lower():
                success_count += 1

    all_facts_recalled = (success_count == len(requested_rewrite))

    multi_hop_accuracy = correct_responses / len(questions)
    edit_success_rate = success_count / len(requested_rewrite)
    instance_accuracy = 1 if all_facts_recalled else 0

    return multi_hop_accuracy, edit_success_rate, instance_accuracy, generated_answers

def main(
    model_name: str,
    ds_name: str,
    dataset_size_limit: int,
    generation_test_interval: int,
    dir_name: str,
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
                    record
                ),
            }

            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)
            print(f"Evaluation took {time() - start} seconds")

if __name__ == "__main__":
    main(
        model_name="EleutherAI/gpt-j-6B",
        ds_name="mquake",
        dataset_size_limit=3000,
        generation_test_interval=1,
        dir_name="your_results_dir"
    )
