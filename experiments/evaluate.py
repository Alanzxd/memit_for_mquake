import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
    MQuAKEDataset_CF_3k,
    MQuAKE_T,
    MQuAKE_2002,
    MQuAKE_hard,
    MQuAKEDataset_CF_3k_A,
    MQuAKE_T_A,
    MQuAKE_2002_A,
    MQuAKE_hard_A,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact, compute_rewrite_quality_mquake
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import MEMITHyperParams, apply_memit_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "mquake_cf_3k": (MQuAKEDataset_CF_3k, compute_rewrite_quality_mquake),
    "mquake_t": (MQuAKE_T, compute_rewrite_quality_mquake),
    "mquake_2002": (MQuAKE_2002, compute_rewrite_quality_mquake),
    "mquake_hard": (MQuAKE_hard, compute_rewrite_quality_mquake),
    "mquake_cf_3k_A": (MQuAKEDataset_CF_3k_A, compute_rewrite_quality_mquake),
    "mquake_t_A": (MQuAKE_T_A, compute_rewrite_quality_mquake),
    "mquake_2002_A": (MQuAKE_2002_A, compute_rewrite_quality_mquake),
    "mquake_hard_A": (MQuAKE_hard_A, compute_rewrite_quality_mquake),
}

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    current_dir = Path.cwd()
    if (
        continue_from_run is None
        or not (run_dir := current_dir / RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = current_dir / RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

        # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        print(model)
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
    
    # 修改模型中的所有 Dropout 层的参数
    def set_dropout(model, p):
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = p
    
    # 将 Dropout 概率设置为 0.5
    set_dropout(model, p=0.5)

    
    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    # Get cache templates
    cache_template = None
    KV_DIR = Path.cwd() / "cache"
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")

    first_batch_model = None
    new_results_dir = run_dir / "first_batch_evaluation" / str(num_edits)
    new_results_dir.mkdir(parents=True, exist_ok=True)

    for i, record_chunks in enumerate(chunks(ds, num_edits)):
        if num_edits == 0:
            break

        case_result_template = str(run_dir / "{}_edits-case_{}.json")

        already_finished = True
        for record in record_chunks:
            if not Path(case_result_template.format(num_edits, record["case_id"])).exists():
                already_finished = False
                break
        if already_finished:
            continue

        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('record', record)
        print('requested rewrite', record["requested_rewrite"])

        flattened_rewrites = [
            {"case_id": record["case_id"], **rw}
            for record in record_chunks
            for rw in record["requested_rewrite"]
        ]

        def apply_algo_with_device(model, tok, rewrites, hparams, **kwargs):
            for rewrite in rewrites:
                input_text = rewrite['prompt'].format(rewrite['subject'])
                input_ids = tok.encode(input_text, return_tensors='pt').to(device)
                attention_mask = input_ids.ne(tok.pad_token_id).to(device)
                rewrite['input_ids'] = input_ids
                rewrite['attention_mask'] = attention_mask
            
            return apply_algo(model, tok, rewrites, hparams, **kwargs)
        
        if num_edits > 0:
            start = time()
            edited_model, weights_copy = apply_algo_with_device(
                model,
                tok,
                flattened_rewrites,
                hparams,
                copy=False,
                return_orig_weights=True,
                **args_conserve_memory,
                **etc_args,
            )
            exec_time = time() - start
            print("Execution took", exec_time)
        else:
            edited_model = model
            exec_time = 0
            weights_copy = None

        if i == 0 and num_edits > 0:
            first_batch_model = edited_model

        print("---------------------------------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------------------------------")
        
        start = time()
        gen_test_vars = [snips, vec]
        for record in record_chunks:
            out_file = Path(case_result_template.format(num_edits, record["case_id"]))
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue

            metrics = {
                "case_id": record["case_id"],
                "grouped_case_ids": case_ids,
                "num_edits": num_edits,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),
                ),
            }

            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)

        if num_edits > 0:
            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda")

        print("Evaluation took", time() - start)
    
    if first_batch_model is not None and num_edits in [1, 100, 1000]:
        print("Evaluating all data with the first batch model...")
        model_save_path = f"{ds_name}_model_{num_edits}_edits"
        print(f"Saving first batch model to {model_save_path}...")
        
        # 创建目录以确保路径存在
        os.makedirs(model_save_path, exist_ok=True)
        
        # 保存模型和tokenizer
        first_batch_model.save_pretrained(model_save_path)
        tok.save_pretrained(model_save_path)
        '''for record_chunks in chunks(ds, num_edits):
            case_result_template = str(new_results_dir / "{}_edits-case_{}.json")
    
            start = time()
            gen_test_vars = [snips, vec]
            for record in record_chunks:
                out_file = Path(case_result_template.format(num_edits, record["case_id"]))
                if out_file.exists():
                    print(f"Skipping {out_file}; already exists")
                    continue
                    
                metrics = {
                    "case_id": record["case_id"],
                    "grouped_case_ids": case_ids,
                    "num_edits": num_edits,
                    "requested_rewrite": record["requested_rewrite"],
                    "time": exec_time,
                    "post": ds_eval_method(
                        first_batch_model,
                        tok,
                        record,
                        num_edits,
                        *(
                            gen_test_vars
                            if record["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),
                    ),
                }
    
                with open(out_file, "w") as f:
                    json.dump(metrics, f, indent=1)
    
            print("Evaluation took", time() - start)'''
        
    
    if num_edits == 0:
        print("Evaluating all data with the original model...")
        for record_chunks in chunks(ds, 1):
            case_result_template = str(run_dir / "{}_edits-case_{}.json")
    
            start = time()
            gen_test_vars = [snips, vec]
            for record in record_chunks:
                out_file = Path(case_result_template.format(num_edits, record["case_id"]))
                if out_file.exists():
                    print(f"Skipping {out_file}; already exists")
                    continue
                exec_time = time() - start
                metrics = {
                    "case_id": record["case_id"],
                    "grouped_case_ids": [record["case_id"]],
                    "num_edits": num_edits,
                    "requested_rewrite": record["requested_rewrite"],
                    "time": exec_time,
                    "post": ds_eval_method(
                        model,
                        tok,
                        record,
                        *(
                            gen_test_vars
                            if record["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),
                    ),
                }
    
                with open(out_file, "w") as f:
                    json.dump(metrics, f, indent=1)
    
            print("Evaluation took", time() - start)


def window(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    if n == 0:
        n = 1
    for i in range(0, len(arr), n):
        yield arr[i : i + n]

if __name__ == "__main__":
    import argparse
    os.environ['HF_HOME'] = '/scratch/xz3645/huggingface_cache'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "ROME", "FT", "MEND"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B", "EleutherAI/gpt-j-6B", "lmsys/vicuna-7b-v1.5"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre", "mquake_cf_3k","mquake_t","mquake_2002","mquake_hard", "mquake_cf_3k_A","mquake_t_A","mquake_2002_A","mquake_hard_A"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
    )

