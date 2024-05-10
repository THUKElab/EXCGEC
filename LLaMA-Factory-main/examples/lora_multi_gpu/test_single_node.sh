#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,6,7 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    ../../src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /data/qins/workspace/zh-exp-llm/exp-cgec/model/Qwen/Qwen1___5-1___8B \
    --dataset  exp_cgec_zh \
    --dataset_dir ../../data \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir base_multi_lora_qwen_checkpoint3 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 3000 \
    --val_size 0.1 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --fp16
