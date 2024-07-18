import os
import json

# 设置json文件的目录路径
directory = "/data/shared/Alan/LLM_Evaluation/memit_for_mquake/results/MEMIT/run_002"

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
        # 打开并读取json文件
        with open(filepath, "r") as file:
            data = json.load(file)
            total_cases += 1
            multi_hop_accuracy = data.get("post", {}).get("multi_hop_accuracy", -1)
            edit_success_rate = data.get("post", {}).get("edit_success_rate", -1)
            instance_accuracy = data.get("post", {}).get("instance_accuracy", -1)

            # 分类
            if multi_hop_accuracy == 1 and edit_success_rate == 1 and instance_accuracy == 1:
                categories["category_1"].append(data["case_id"])
            elif 0 < multi_hop_accuracy < 1 and edit_success_rate == 1 and instance_accuracy == 1:
                categories["category_2"].append(data["case_id"])
            elif multi_hop_accuracy == 0 and edit_success_rate == 1 and instance_accuracy == 1:
                categories["category_3"].append(data["case_id"])
            elif multi_hop_accuracy == 1 and 0 < edit_success_rate < 1 and instance_accuracy == 0:
                categories["category_4"].append(data["case_id"])
            elif 0 < multi_hop_accuracy < 1 and 0 < edit_success_rate < 1 and instance_accuracy == 0:
                categories["category_5"].append(data["case_id"])
            elif multi_hop_accuracy == 0 and 0 < edit_success_rate < 1 and instance_accuracy == 0:
                categories["category_6"].append(data["case_id"])
            elif multi_hop_accuracy == 0 and edit_success_rate == 0 and instance_accuracy == 0:
                categories["category_7"].append(data["case_id"])
            elif multi_hop_accuracy > 0 and edit_success_rate == 0 and instance_accuracy == 0:
                categories["category_8"].append(data["case_id"])

# 计算各类的比例
for category, case_ids in categories.items():
    proportion = len(case_ids) / total_cases if total_cases > 0 else 0
    categories[category] = {
        "case_ids": case_ids,
        "proportion": proportion
    }

# 将结果存储到一个json文件中
output_filepath = "/data/shared/Alan/LLM_Evaluation/memit_for_mquake/results/MEMIT/run_002_classification.json"
with open(output_filepath, "w") as output_file:
    json.dump(categories, output_file, indent=4)

# 打印结果
for category, details in categories.items():
    print(f"{category} - {len(details['case_ids'])} cases, proportion: {details['proportion']:.2%}")

print(f"Total cases: {total_cases}")
