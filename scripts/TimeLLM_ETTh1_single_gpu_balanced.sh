#!/bin/bash
# TimeLLM ETTh1 Single GPU Training Script - BALANCED VERSION
# Optimized for faster training while maintaining convergence

model_name=TimeLLM
train_epochs=30  # Reduced from 100 but more than ultra-fast 10
learning_rate=0.001  # Much lower than ultra-fast 0.02, but reasonable
llama_layers=24  # Compromise between 32 (full) and 16 (ultra-fast)

master_port=00097
num_process=1
batch_size=16  # Keep larger batch size for efficiency
d_model=32
d_ff=128

comment='TimeLLM-ETTh1-SingleGPU-4bit-BALANCED'

# WandB configuration
use_wandb=true
wandb_project="TimeLLM-Experiments"
wandb_entity=""
wandb_run_name="ETTh1-SingleGPU-4bit-BALANCED-$(date +%Y%m%d_%H%M%S)"

echo "Starting balanced training with convergence-friendly settings..."
echo "Key differences from ultra-fast:"
echo "  - Learning rate: 0.001 (vs 0.02 ultra-fast)"
echo "  - Epochs: 30 (vs 10 ultra-fast)"
echo "  - LLaMA layers: 24 (vs 16 ultra-fast)"
echo "  - Sequence length: 512 (vs 256 ultra-fast)"
echo "  - Patience: 10 (vs 5 ultra-fast)"

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
  --patience 10 \
  --use_wandb \
  --wandb_project $wandb_project \
  --wandb_run_name "${wandb_run_name}_pred96" \
  $([ -n "$wandb_entity" ] && echo "--wandb_entity $wandb_entity" || echo "")

echo "Training completed! This balanced approach should converge properly."