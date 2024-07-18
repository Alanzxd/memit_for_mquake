import os
import json
import sys

def analyze_errors(run_number):
    # 设置json文件的目录路径
    directory = f"/data/shared/Alan/LLM_Evaluation/memit_for_mquake/results/MEMIT/run_{run_number:03d}"

    # 初始化分类字典
    categories = {
        "category_1": [], # muti=1，edit=1，instance=1
        "category_2": [], # muti大于0小于1，edit=1，instance=1
        "category_3": [], # muti等于0，edit=1，instance=1
        "category_4": [], # muti等于1，edit大于0小于1，instance=0
        "category_5": [], # muti大于0小于1，edit大于0小于1，instance=0
        "category_6": [], # muti等于0，edit大于0小于1，instance=0
        "category_7": [], # muti等于0，edit=0，instance=0
        "category_8": []  # muti大于0，edit=0，instance=0
    }

    # 统计总数
    total_cases = 0

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            # 构造文件的完整路径
            filepath = os.path.join(directory, filename)
            try:
                # 打开并读取json文件
                with open(filepath, "r") as file:
                    content = file.read().strip()
                    if not content:
                        print(f"Warning: {filename} is empty.")
                        continue

                    data = json.loads(content)
                    total_cases += 1
                    multi_hop_accuracy = data.get("post", {}).get("multi_hop_accuracy", -1)
                    edit_success_rate = data.get("post", {}).get("edit_success_rate", -1)
                    instance_accuracy = data.get("post", {}).get("instance_accuracy", -1)

                    # 分类
                    if multi_hop_accuracy == 1 and edit_success_rate == 1 and instance_accuracy == 1:
                        categories["category_1"].append(data)
                    elif 0 < multi_hop_accuracy < 1 and edit_success_rate == 1 and instance_accuracy == 1:
                        categories["category_2"].append(data)
                    elif multi_hop_accuracy == 0 and edit_success_rate == 1 and instance_accuracy == 1:
                        categories["category_3"].append(data)
                    elif multi_hop_accuracy == 1 and 0 < edit_success_rate < 1 and instance_accuracy == 0:
                        categories["category_4"].append(data)
                    elif 0 < multi_hop_accuracy < 1 and 0 < edit_success_rate < 1 and instance_accuracy == 0:
                        categories["category_5"].append(data)
                    elif multi_hop_accuracy == 0 and 0 < edit_success_rate < 1 and instance_accuracy == 0:
                        categories["category_6"].append(data)
                    elif multi_hop_accuracy == 0 and edit_success_rate == 0 and instance_accuracy == 0:
                        categories["category_7"].append(data)
                    elif multi_hop_accuracy > 0 and edit_success_rate == 0 and instance_accuracy == 0:
                        categories["category_8"].append(data)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {filename}")

    # 计算各类的比例
    for category, cases in categories.items():
        proportion = len(cases) / total_cases if total_cases > 0 else 0
        categories[category] = {
            "cases": cases,
            "proportion": proportion
        }

    # 将结果存储到一个json文件中
    output_filepath = f"/data/shared/Alan/LLM_Evaluation/memit_for_mquake/results/MEMIT/run_{run_number:03d}_classification.json"
    with open(output_filepath, "w") as output_file:
        json.dump(categories, output_file, indent=4)

    # 打印结果
    for category, details in categories.items():
        print(f"{category} - {len(details['cases'])} cases, proportion: {details['proportion']:.2%}")

    print(f"Total cases: {total_cases}")

    return categories

def create_new_dataset(categories, run_number, source_file, output_file):
    output_answers_file = f"/data/shared/Alan/LLM_Evaluation/memit_for_mquake/results/MEMIT/run_{run_number:03d}_answers.json"

    # 读取原始数据集
    with open(source_file, "r") as file:
        data = json.load(file)

    # 初始化新的数据集和答案集
    new_dataset = []
    answers_dataset = []

    # 从每个类别中各截取最多20个例子
    for category, details in categories.items():
        case_ids = [case["case_id"] for case in details["cases"]]
        examples = [case for case in data if case["case_id"] in case_ids][:20]
        
        # 添加类型字段并扩展数据集
        for example in examples:
            example["type"] = category
            new_dataset.append(example)

        # 记录回答
        for case in details["cases"][:20]:
            answers_dataset.append({
                "case_id": case["case_id"],
                "grouped_case_ids": case.get("grouped_case_ids", []),
                "num_edits": case.get("num_edits", 0),
                "requested_rewrite": case.get("requested_rewrite", []),
                "time": case.get("time", 0),
                "post": case.get("post", {})
            })

    # 将新的数据集写入文件
    with open(output_file, "w") as file:
        json.dump(new_dataset, file, indent=4)

    # 将回答数据集写入文件
    with open(output_answers_file, "w") as file:
        json.dump(answers_dataset, file, indent=4)

    print(f"New dataset created with {len(new_dataset)} examples and saved to {output_file}")
    print(f"Answers dataset created and saved to {output_answers_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python error_analysis.py <run_number> <source_file> <output_file>")
    else:
        run_number = int(sys.argv[1])
        source_file = sys.argv[2]
        output_file = sys.argv[3]
        categories = analyze_errors(run_number)
        create_new_dataset(categories, run_number, source_file, output_file)
