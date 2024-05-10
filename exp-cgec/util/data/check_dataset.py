import json

def process_json_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    id = 0
    output_list = []
    for sample in data:
        print("1")
        output_data = json.loads(sample["output"])
        print(output_data)
        sample["output"] = output_data
        output_list.append(sample)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_list, f, indent=4, ensure_ascii=False)

# 使用示例
input_file = "/data/qins/workspace/zh-exp-llm/exp-cgec/util/data/test1.json"
output_file = "/data/qins/workspace/zh-exp-llm/exp-cgec/data/test1.json"
process_json_file(input_file, output_file)