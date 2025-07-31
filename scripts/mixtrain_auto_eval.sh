#!/bin/bash

#############################################
# Part 1: Training
#############################################

PROMPT_VERSION="llama3"
WANDB_PROMPT_VERSION="Deepseek-MoE"
MODEL_VERSION="llama"

GRAPH_TOWER="moleculestm"
INIT_CHECKPOINT_GNN="assets/moleculestm.pth"

CHECKPOINT_FOLDER_PREFIX="_checkpoints"
TASK="forward:1/retrosynthesis:1/reagent:1/homolumo:1/molcap:1/solvent:1/catalyst:1/yield:1"
# TASK="forward:1/retrosynthesis:1/reagent:1/homolumo:1/molcap:1/solvent:1/catalyst:1/yield:1/experiment:1/tpsa:1/weight:1/dqa:1/logp:1/iupac:1/textguidedmolgen:1/molediting:1"
PROJECTOR="naive_linear"
REMARK="1B-deepseek-moe-5EP-qurater-sharedEP-clip-alpha-embed-Tok2-8tasks"

export WANDB_ENTITY="Omni-Mol"
export WANDB_PROJECT="${WANDB_ENTITY}_${WANDB_PROMPT_VERSION}"
export WANDB_API_KEY="ba70fcbc92808cc7a1750dd80ac3908295e6854f"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========== Start Training =========="
# --deepspeed scripts/zero_configs/zero2.json \
deepspeed --master_port 29505 train.py \
    --training_recipe loramoe \
    --use_alpha True \
    --task_config $TASK \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --language_backbone_name $MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path data/train \
    --data_type $TASK \
    --graph_tower $GRAPH_TOWER \
    --mm_projector_type $PROJECTOR \
    --graph_init_checkpoint $INIT_CHECKPOINT_GNN \
    --pretrain_mm_mlp_adapter assets/mm_projector.bin \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/$MODEL_VERSION-$REMARK \
    --num_train_epochs 15 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 1 \
    --stop_epoch 12 \
    --eval_strategy "no" \
    --eval_steps 500 \
    --split_eval False \
    --val_ratio 0.1 \
    --eval_on_start False \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 8e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.0075 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to all \
    --logging_dir tf-logs/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --moe_class deepseek \
    --moe_mode second_quarter\
    --ep_size 1 \
    --num_experts 5 \
    --use_residual True \
    --router_aux_loss_coef 0.01 \
    --is_training True \
    --top_k_experts 2 \
    --use_task_loss False \
    --ddp_find_unused_parameters False \
    --norm_topk_prob False

echo "========== Training Finished =========="


#############################################
# Part 2: Evaluation (Test)
#############################################

echo "========== Start Evaluation =========="

accelerate launch --main_process_port 29533 auto_eval.py \
    --save_path eval_results/$MODEL_VERSION-$REMARK/ \
    --device cuda \
    --temperature 0.2 \
    --num_beams 1 \
    --max_new_tokens 512 \
    --repetition_penalty 1.0 \
    --metric_path True \
    --model_type lora+moe \
    --prompt_version llama3 \
    --model_path $CHECKPOINT_FOLDER_PREFIX/$MODEL_VERSION-$REMARK \
    --language_backbone meta-llama/Llama-3.2-1B-Instruct \
    --graph_path assets/moleculestm.pth \
    --use_flash_attn True \
    --task_embed False \
    --dtype bfloat16 \
    --data_path data/evaluate \
    --add_selfies True \


echo "========== All Evaluation Finished =========="
