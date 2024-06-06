import collections
import json
from pprint import pprint
from typing import List, Optional

import numpy as np

from util.globals import *

def main(
    dir_name,
    runs: Optional[List],
    first_n_cases=None,
    get_uncompressed=False,
    abs_path=False,
):
    summaries = []
    uncompressed = []

    total_correct_cases_any = 0  # 记录至少有一个问题正确的案例数
    total_correct_cases_all = 0  # 记录所有问题都正确的案例数
    total_cases = 0  # 记录总案例数
    total_correct_questions = 0  # 记录总正确问题数
    total_edit_success_rate = 0  # 记录总的edit-wise success rate
    total_instance_accuracy = 0  # 记录总的instance-wise accuracy

    for run_dir in (RESULTS_DIR / dir_name if not abs_path else dir_name).iterdir():
        # Skip if we're not interested
        if runs is not None and all(run not in str(run_dir) for run in runs):
            continue

        # Iterate through all case files
        cur_sum = collections.defaultdict(lambda: [])
        files = list(run_dir.glob("*case_*.json"))
        files.sort(key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
        for case_file in files:
            try:
                with open(case_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Could not decode {case_file} due to format error; skipping.")
                continue

            case_id = data["case_id"]
            if first_n_cases is not None and case_id >= first_n_cases:
                break

            if "time" in data:
                cur_sum["time"].append(data["time"])

            post_data = data.get('post', {})
            multi_hop_acc = post_data.get('multi_hop_accuracy', None)
            edit_success_rate = post_data.get('edit_success_rate', None)
            instance_accuracy = post_data.get('instance_accuracy', None)

            if multi_hop_acc is not None:
                cur_sum["multi_hop_accuracy"].append(multi_hop_acc)
                total_cases += 1
                if multi_hop_acc > 0:
                    total_correct_cases_any += 1
                    if multi_hop_acc == 1:
                        total_correct_cases_all += 1
                total_correct_questions += multi_hop_acc * 3

            if edit_success_rate is not None:
                total_edit_success_rate += edit_success_rate

            if instance_accuracy is not None:
                total_instance_accuracy += instance_accuracy

        if len(cur_sum) == 0:
            continue

        num_items = len(cur_sum[next(iter(cur_sum.keys()))])
        metadata = {
            "run_dir": str(run_dir),
            "num_cases": num_items,
        }

        uncompressed.append(dict(cur_sum, **metadata))

        cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}
        for k, v in cur_sum.items():
            if all(exclude not in k for exclude in ["essence_score", "time"]):
                # Constant multiplication scales linearly with mean and stddev
                cur_sum[k] = tuple(np.around(z * 100, 2) for z in v)

        cur_sum.update(metadata)
        pprint(cur_sum)
        summaries.append(cur_sum)

    # 计算并输出各种多跳准确率
    print(f"Total Multi-hop Accuracy (per case, any question correct): {total_correct_cases_any / total_cases if total_cases > 0 else 0}")
    print(f"Total Multi-hop Accuracy (per case, all questions correct): {total_correct_cases_all / total_cases if total_cases > 0 else 0}")
    print(f"Total Multi-hop Accuracy (per question): {total_correct_questions / (total_cases * 3) if total_cases > 0 else 0}")

    # 计算并输出平均 edit-wise success rate 和 instance-wise accuracy
    print(f"Average Edit-wise Success Rate: {total_edit_success_rate / total_cases if total_cases > 0 else 0}")
    print(f"Average Instance-wise Accuracy: {total_instance_accuracy / total_cases if total_cases > 0 else 0}")

    return uncompressed if get_uncompressed else summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", type=str, help="Name of directory to scan for runs."
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="By default, summarizes each run in <dir_name>. "
        "If runs are specified, only evaluates those specific runs.",
    )
    parser.add_argument(
        "--first_n_cases",
        type=int,
        default=None,
        help="Restricts evaluation to first n cases in dataset. "
        "Useful for comparing different in-progress runs on the same slice of data.",
    )
    args = parser.parse_args()

    main(
        args.dir_name,
        None if args.runs is None else args.runs.split(","),
        args.first_n_cases,
    )
