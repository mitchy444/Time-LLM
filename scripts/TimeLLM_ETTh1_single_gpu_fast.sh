#!/bin/bash
# TimeLLM ETTh1 Single GPU Training Script - FAST VERSION
# Optimized for speed while maintaining reasonable performance

model_name=TimeLLM
train_epochs=20  # Reduced from 100 - should cut training time by 80%
learning_rate=0.01
llama_layers=32

master_port=00097
num_process=1
# Increased batch size for faster training (you mentioned using 12, let's try 16-24)
batch_size=16
d_model=32
d_ff=128

comment='TimeLLM-ETTh1-SingleGPU-4bit-FAST'

# WandB configuration
use_wandb=true
wandb_project="TimeLLM-Experiments"
wandb_entity=""  # Leave empty or set your wandb username/team
wandb_run_name="ETTh1-SingleGPU-4bit-FAST-$(date +%Y%m%d_%H%M%S)"

# Only train on one prediction length first to test (96 steps)
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
  --wandb_run_name "${wandb_run_name}_pred96" \
  $([ -n "$wandb_entity" ] && echo "--wandb_entity $wandb_entity" || echo "")