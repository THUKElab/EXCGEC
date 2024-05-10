#qwen18_dir="/data/qins/workspace/zh-exp-llm/exp-cgec/model/ft_json_chat_single_qwen_1.5"
##qwen18_dir="/data/qins/workspace/zh-exp-llm/exp-cgec/model/Qwen/Qwen1___5-1___8B"
#from transformers import AutoModelForCausalLM, AutoTokenizer
#device = "cuda" # the device to load the model onto
#
#model = AutoModelForCausalLM.from_pretrained(
#    qwen18_dir,
#    torch_dtype="auto",
#    device_map="auto"
#)
#tokenizer = AutoTokenizer.from_pretrained(qwen18_dir)
#
##special_tokens_dict = {'additional_special_tokens': ['<eos_target>','<ST>', '<TT>', '<T>', '<S>','<eos_explanation>']}
##num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
##print('We have added', num_added_toks, 'tokens')
##model.resize_token_embeddings(len(tokenizer))
#
#prompt = "我是复旦大学国际政治系（国际关系专业毕业的）。"
#messages = [
#    {"role": "system", "content": "将以下文本进行语法纠错并生成纠正后的句子以及纠正相关的解释信息"},
#    {"role": "user", "content": prompt}
#]
#text = tokenizer.apply_chat_template(
#    messages,
#    tokenize=False,
#    add_generation_prompt=True
#)
#model_inputs = tokenizer([text], return_tensors="pt").to(device)
#
#generated_ids = model.generate(
#    model_inputs.input_ids,
#    max_new_tokens=512
#)
#generated_ids = [
#    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#]
#
#response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
#
#print(response)

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def predict(input_file, output_file, model, tokenizer):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_list = []
    for sample in data:
        prompt = sample["input"]
        messages = [
            {"role": "system", "content": "将以下文本进行语法纠错并生成纠正后的句子以及纠正相关的解释信息"},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

        output_sample = {
            'input': prompt,
            'output': response
        }
        output_list.append(output_sample)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_list, f, indent=4, ensure_ascii=False)

# 加载模型和分词器
qwen18_dir = "/data/qins/workspace/zh-exp-llm/exp-cgec/model/ft_json_chat_single_qwen_1.5"
device = "cuda"  # 模型加载的设备

model = AutoModelForCausalLM.from_pretrained(
    qwen18_dir,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(qwen18_dir)

# 预测并保存结果
input_file = "/data/qins/workspace/zh-exp-llm/exp-cgec/util/data/test.json"
output_file = "./output/ft_json_chat_single_qwen15.json"
predict(input_file, output_file, model, tokenizer)