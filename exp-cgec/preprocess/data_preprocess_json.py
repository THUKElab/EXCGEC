import json

def process_json_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_list = []
    for sample in data["samples"]:
        #idx = sample['index']
        source = sample['source']
        target = sample['target']
        edits = sample['edits']
        
        output_edits = []
        for edit in edits:
            #tgt_idx = edit['tgt_idx']
            src_tokens = edit['src_tokens']
            tgt_tokens = edit['tgt_tokens']
            error_type = edit['error_type']
            error_severity = edit['error_severity']
            error_description = edit['error_description']
            
            edit_str = f"{{\"src_tokens\": \"{src_tokens}\", \"tgt_tokens\": \"{tgt_tokens}\", \"error_type\": \"{error_type}\", \"error_severity\": \"{error_severity}\", \"error_description\": \"{error_description}\"}}"
            output_edits.append(edit_str)
        
        output_sample = {
            "instruction": instruction,
            'input': source,
            'output': f"{{\"target\": \"{target}\", \"edits\": [{', '.join(output_edits)}]}}"
        }
        
        output_list.append(output_sample)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_list, f, indent=4, ensure_ascii=False)

# 使用示例
data_path = "../data/"
input_file = data_path + "XCGEC_202400429.json"
output_file = data_path + "input.json"
instruction = "将以下文本进行语法纠错并生成纠正后的句子以及纠正相关的解释信息"
process_json_file(input_file, output_file)