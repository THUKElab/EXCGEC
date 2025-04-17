import json

data_path = "../data/"
input_file = data_path + "XCGEC_202400429.json"
instruction = "将以下文本进行语法纠错并生成纠正后的句子以及纠正相关的解释信息"

with open(input_file, 'r') as file:
    data = json.load(file)

output_list = []

for sample in data["samples"]:
    index = sample["index"]
    source = sample["source"]
    target = sample["target"]
    edits = sample["edits"]

    edit_dict = ""
    for idx, edit in enumerate(edits):
        tgt_idx = edit["tgt_idx"]
        src_tokens = edit["src_tokens"]
        tgt_tokens = edit["tgt_tokens"]
        error_type = edit["error_type"]
        error_severity = edit["error_severity"]
        error_description = edit["error_description"]
        if len(src_tokens) == 0:
            src_tokens = "<emp>"
        if len(tgt_tokens) == 0:
            tgt_tokens = "<emp>"

        edit_dict += f"{src_tokens} <ST> "+f"{tgt_tokens} <TT> "f"{error_type} <T> "f"{error_severity} <S> "
        if idx == len(edits):
            edit_dict += f"{error_description} <eos_explanation> <eos> "
        else:
            edit_dict += f"{error_description} <eos_explanation> "
        #output_list.append(edit_dict)
    
    output_dict = {
        "instruction": instruction,
        "input": source,
        "output": f"{target} <eos_target> " + edit_dict
    }
    
    output_list.append(output_dict)

output_file = data_path + "input.json"

with open(output_file, 'w') as file:
    json.dump(output_list, file, indent=4, ensure_ascii=False)