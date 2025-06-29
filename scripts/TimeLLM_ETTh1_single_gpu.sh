#!/bin/bash
# TimeLLM ETTh1 Single GPU Training Script with 4-bit Quantization
# This script runs TimeLLM with 4-bit quantization enabled (hardcoded in models/TimeLLM.py)
# 4-bit quantization reduces memory usage by ~75% while maintaining model performance

model_name=TimeLLM
train_epochs=100
learning_rate=0.01
llama_layers=32

master_port=00097
num_process=1
# Optimized batch sizes for 4-bit quantization (can use larger batches due to reduced memory usage)
batch_size=32
d_model=32
d_ff=128

comment='TimeLLM-ETTh1-SingleGPU-4bit'

# WandB configuration
use_wandb=true
wandb_project="TimeLLM-Experiments"
wandb_entity=""  # Leave empty or set your wandb username/team
wandb_run_name="ETTh1-SingleGPU-4bit-$(date +%Y%m%d_%H%M%S)"

# Optimized for 4-bit quantization: using bf16 mixed precision for better performance
accelerate launch --mixed_precision bf16 --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --use_wandb \
  --wandb_project $wandb_project \
  --wandb_entity $wandb_entity \
  --wandb_run_name "${wandb_run_name}_pred96"

accelerate launch --mixed_precision bf16 --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 32 \
  --d_ff 128 \
  --batch_size $batch_size \
  --learning_rate 0.02 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --use_wandb \
  --wandb_project $wandb_project \
  --wandb_entity $wandb_entity \
  --wandb_run_name "${wandb_run_name}_pred192"

accelerate launch --mixed_precision bf16 --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --lradj 'COS'\
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --use_wandb \
  --wandb_project $wandb_project \
  --wandb_entity $wandb_entity \
  --wandb_run_name "${wandb_run_name}_pred336"

accelerate launch --mixed_precision bf16 --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --use_wandb \
  --wandb_project $wandb_project \
  --wandb_entity $wandb_entity \
  --wandb_run_name "${wandb_run_name}_pred720"