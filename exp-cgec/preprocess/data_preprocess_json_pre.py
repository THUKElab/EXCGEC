import json
import re

def process_json_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_list = []
    for sample in data["samples"]:
        source = sample['source']
        target = sample['target']
        edits = sample['edits']
        
        output_edits = []
        output_exps = []
        for edit in edits:
            edit_str = {
                "src_interval": edit['src_interval'],
                "tgt_interval": edit['tgt_interval'],
                "src_tokens": edit['src_tokens'],
                "tgt_tokens": edit['tgt_tokens']
            }
            
            exp_str = {
                "error_type": edit['error_type'],
                "error_severity": edit['error_severity'],
                "error_description": edit['error_description']
            }
            
            output_edits.append(edit_str)
            output_exps.append(exp_str)
            
        
        output = {
            "edits": output_edits,
            "explanations": output_exps,
            "target": target  
        }
        
        output_str = json.dumps(output, ensure_ascii=False)
        output_str = re.sub(r'"target"', r'<TGT>"target"', output_str)
        output_sample = {
            "instruction": instruction,
            "input": source,
            #"output": f"{{\"target\": \"{target}\", <TGT>\"edits\": [{', '.join(output_edits)}], \"explanations\": [{', '.join(output_exps)}]}}"
            "output": output_str
        }
        
        output_list.append(output_sample)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_list, f, indent=4, ensure_ascii=False)

# 使用示例
data_path = "../data/splits/"
input_file = data_path + "test.json"
output_file = data_path + "test_out_pre.json"
instruction = "将以下文本进行语法纠错并生成纠正后的句子以及纠正相关的解释信息"
process_json_file(input_file, output_file)