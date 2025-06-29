#!/bin/bash
# TimeLLM ETTh1 Single GPU Training Script - ULTRA FAST VERSION
# Maximum speed optimizations - should reduce 133 hours to ~3-6 hours

model_name=TimeLLM
train_epochs=10  # Reduced from 100 - 90% time reduction
learning_rate=0.02  # Higher learning rate for faster convergence
llama_layers=16  # Reduced from 32 - 50% model size reduction

master_port=00097
num_process=1
# Larger batch size for faster training (adjust based on your GPU memory)
batch_size=24
d_model=32
d_ff=128

comment='TimeLLM-ETTh1-SingleGPU-4bit-ULTRAFAST'

# WandB configuration
use_wandb=true
wandb_project="TimeLLM-Experiments"
wandb_entity=""
wandb_run_name="ETTh1-SingleGPU-4bit-ULTRAFAST-$(date +%Y%m%d_%H%M%S)"

# Train only the most important prediction length (96) first
# You can add others later if needed
accelerate launch --mixed_precision bf16 --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 256 \
  --label_len 24 \
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
  --patience 5 \
  --use_wandb \
  --wandb_project $wandb_project \
  --wandb_run_name "${wandb_run_name}_pred96" \
  $([ -n "$wandb_entity" ] && echo "--wandb_entity $wandb_entity" || echo "")

echo "Training completed! Check results before running additional prediction lengths."