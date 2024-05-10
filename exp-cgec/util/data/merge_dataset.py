import json
import random

def extract_entries(input_file, output_file, num_entries=10):
    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 检查是否有足够的条目可供抽取
    if len(data) < num_entries:
        raise ValueError("The number of entries in the file is less than the number to extract.")
    
    # 随机选择条目
    selected_entries = random.sample(data, num_entries)
    
    # 写入新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(selected_entries, file, ensure_ascii=False, indent=4)
    
    # 从原始数据中删除这些条目
    remaining_entries = [entry for entry in data if entry not in selected_entries]
    
    # 将更新后的数据写回原始文件
    with open(input_file, 'w', encoding='utf-8') as file:
        json.dump(remaining_entries, file, ensure_ascii=False, indent=4)

# 使用函数
input_file = '/data/qins/workspace/zh-exp-llm/LLaMA-Factory-main/data/exp_cgec_qwen1.5_data_zh.json'  # 原始文件的路径
output_file = './test.json'  # 输出文件的路径

extract_entries(input_file, output_file)
