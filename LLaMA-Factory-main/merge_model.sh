#!/bin/bash
python src/export_model.py \
    --model_name_or_path /data/qins/workspace/zh-exp-llm/exp-cgec/model/Qwen/Qwen1___5-1___8B-Chat \
    --adapter_name_or_path /data/qins/workspace/zh-exp-llm/LLaMA-Factory-main/chat_json_single_lora_qwen_checkpoint3 \
    --template qwen \
    --finetuning_type lora \
    --export_dir  /data/qins/workspace/zh-exp-llm/exp-cgec/model/ft_json_chat_single_qwen_1.5/ \
    --export_size 2 \
    --export_legacy_format false