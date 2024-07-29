import os
import json

# 设置相对路径
base_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs_n5_t10", "ER"
                                , "have_answer")

for filename in os.listdir(base_folder_path):
    if "link" not in filename:
        continue
    path = base_folder_path + '/' + filename
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if 'instruction' in item:
            item['instruction'] = item['instruction'].replace('direct', 'temporal')

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
