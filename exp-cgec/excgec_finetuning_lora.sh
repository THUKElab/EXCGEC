#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

cd ../LLaMA-Factory/

MODEL_PATH="/data/LLMs/deepseek-ai/deepseek-llm-7b-chat"
TRAIN_DATASET="release_train_exp_cgec"
VALID_DATASET="release_valid_exp_cgec"
TEMPLATE="deepseek"
OUTPUT_DIR="./model/${TEMPLATE}-llm-7b-chat_single_lora_checkpoint3"
EXPORT_DIR=" ../LLM/${TEMPLATE}-llm-7b-chat_single_lora_checkpoint3"
input_file="./data/splits/test_out.json"
output_file="./output/output.json"
LOG_FILE="./log/log.txt"
filepath_hyp="./output/json/output.json"
filepath_ref="./data/splits/test_out_check_fin.json"


######### Training #########
echo "######### Training #########" >> $LOG_FILE
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path ${MODEL_PATH} \
    --dataset ${TRAIN_DATASET},${VALID_DATASET} \
    --template ${TEMPLATE} \
    --lora_target q_proj,v_proj \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --finetuning_type lora \
    --plot_loss \
    --val_size 0.1116 \
    --fp16 \
    --new_special_tokens "<TGT>" \
    --resize_vocab True \
    --lora_rank 8 \
    >> $LOG_FILE 2>&1

######### Export Model #########
echo "######### Exporting Model #########" >> $LOG_FILE
CUDA_VISIBLE_DEVICES=0 python src/export_model.py \
    --model_name_or_path ${MODEL_PATH} \
    --adapter_name_or_path ${OUTPUT_DIR}  \
    --template ${TEMPLATE} \
    --finetuning_type lora \
    --export_dir  ${EXPORT_DIR} \
    --export_size 2 \
    --new_special_tokens "<TGT>" \
    --export_legacy_format false
    >> $LOG_FILE 2>&1 \

cd ../exp-cgec
######### Prediction #########
LOG_FILE="../LLaMA-Factory/log/log.txt"
echo "######### Running Prediction #########" >> $LOG_FILE
CUDA_VISIBLE_DEVICES=0 python predict.py \
    --input_file ${input_file} \
    --output_file ${output_file} \
    --model_dir ${EXPORT_DIR} \
    >> $LOG_FILE 2>&1

######### Data-process #########
echo "######### Data-process #########" >> $LOG_FILE
CUDA_VISIBLE_DEVICES=0 python ./util/data/data-process.py \
    --input_file ${output_file} \
    --output_file ${filepath_hyp} \
    >> $LOG_FILE 2>&1


######### Evaluation #########
echo "######### Running Evaluation #########" >> $LOG_FILE
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate excgec-eval
python evaluation.py \
    --filepath_hyp ${filepath_hyp} \
    --filepath_ref ${filepath_ref} \
    >> $LOG_FILE 2>&1 

conda deactivate

