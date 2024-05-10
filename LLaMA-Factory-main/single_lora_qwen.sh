#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"                              

CUDA_VISIBLE_DEVICES=4 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/qins/workspace/zh-exp-llm/exp-cgec/model/Qwen/Qwen1___5-1___8B \
    --dataset exp_cgec_zh \
    --template qwen \
    --lora_target q_proj,v_proj \
    --output_dir base_json_single_lora_qwen_checkpoint3 \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --finetuning_type lora \
    --fp16 \
    --lora_rank 8