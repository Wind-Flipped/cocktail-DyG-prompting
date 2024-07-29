import os
import json
from collections import Counter
import pandas
import pandas as pd


def preprocess_data(output):
    # 删除前缀 "Answer: " 和 "["
    output = output.replace("Answer: ", "").replace("[", "").replace("]", "")
    # 删除所有空格
    output = output.replace(" ", "").replace("\n", "").replace("\t", "").replace("assistant<|end_header_id|>","")
    return output


def match_answer(truth, answer, task):
    if answer is None:
        return False
    if "What neighbors at time" in task or "What neighbors in periods" in task:
        truth = preprocess_data(truth)
        answer = preprocess_data(answer)
        return Counter(truth) == Counter(answer)
    elif "Find temporal path" in task:
        truth = truth.replace("Answer: ", "").replace(" ", "").replace("\n", "").replace("\t", "")
        answer = answer.replace("Answer: ", "").replace(" ", "").replace("\n", "").replace("\t", "")
        return answer in truth
    else:
        return preprocess_data(truth) == preprocess_data(answer)


def calculate_accuracy_for_file(file_path):
    total_items = 0
    correct_items = {
        "zero_shot": 0,
        "one_shot": 0,
    }

    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        # 遍历每个项目
        for item in data:
            truth = item.get("truth")
            if match_answer(truth, item.get("one_shot_response"), item.get("task")):
                correct_items["one_shot"] += 1
            if match_answer(truth, item.get("zero_shot_output"), item.get("task")):
                correct_items["zero_shot"] += 1
            total_items += 1

    # 计算准确率

    return correct_items["zero_shot"] / total_items, correct_items["one_shot"] / total_items


def calculate_accuracy_for_all_files(base_folder_path):
    results = []

    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(base_folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(base_folder_path, filename)

            # 计算每个文件的准确率
            zero, one = calculate_accuracy_for_file(file_path)
            results.append((filename, zero, one))

    return results


function = ("EXPLAIN")
graph = "graphs_n5_t10"
model = ("Llama3.1")
# 设置相对路径
base_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"output", model, graph, "ER"
                                , function, "have_answer")


# 计算并输出每个文件的准确率
results = calculate_accuracy_for_all_files(base_folder_path)
df = pd.DataFrame(results)
df.to_csv("result_have_answer.csv")
print("HAVE_ANSWER")
for filename, zero, one in results:
    print(
        f"File: {filename:<50} Accuracy: zero: {zero * 100:6.2f}%  one: {one * 100:6.2f}%")

# 设置相对路径
base_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", model, graph, "ER"
                                , function, "no_answer")

print("")
results = calculate_accuracy_for_all_files(base_folder_path)
df = pd.DataFrame(results)
df.to_csv("result_no_answer.csv")
# 计算并输出每个文件的准确率

for filename, zero, one in results:
    print(
        f"File: {filename:<50} Accuracy: zero: {zero * 100:6.2f}%  one: {one * 100:6.2f}%")
