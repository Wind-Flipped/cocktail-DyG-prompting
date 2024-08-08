import argparse
import os
import json
from collections import Counter
import pandas
import pandas as pd


def preprocess_data(output):

    output = output.replace("Answer: ", "").replace("[", "").replace("]", "")

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
        # return preprocess_data(truth) == preprocess_data(answer)
        return truth in answer


def calculate_accuracy_for_file(file_path):
    total_items = 0
    correct_items = {
        "zero_shot": 0,
        # "one_shot": 0,
    }


    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        for item in data:
            truth = item.get("truth")
            if match_answer(truth, item.get("zero_shot_output"), item.get("task")):
                correct_items["zero_shot"] += 1
            total_items += 1



    return correct_items["zero_shot"] / total_items


def calculate_accuracy_for_all_files(base_folder_path):
    results = []


    for filename in os.listdir(base_folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(base_folder_path, filename)

            zero= calculate_accuracy_for_file(file_path)
            results.append((filename, zero))

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="prompting")

    parser.add_argument("--COT", default="NO", help="select the prompting")

    parser.add_argument("-m", "--model_name", default="Llama3.1", help="Enter your model name")

    parser.add_argument("--api_key", default=" ", help="Enter your model name")

    args = parser.parse_args()
    # TODO
    function = (args.COT)
    graph = "graphs_n5_t10"
    model = (args.model_name)


    base_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"output", model, graph, "ER"
                                    , function, "have_answer")


    results = calculate_accuracy_for_all_files(base_folder_path)
    df = pd.DataFrame(results)
    df.to_csv("result_have_answer.csv")
    print("HAVE_ANSWER")
    for filename, zero in results:
        print(
            f"File: {filename:<50} Accuracy: zero: {zero * 100:6.2f}%")


    base_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", model, graph, "ER"
                                    , function, "no_answer")

    print("")
    results = calculate_accuracy_for_all_files(base_folder_path)
    df = pd.DataFrame(results)
    df.to_csv("result_no_answer.csv")


    for filename, zero in results:
        print(
            f"File: {filename:<50} Accuracy: zero: {zero * 100:6.2f}%")
