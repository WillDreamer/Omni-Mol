#!/bin/bash

# T5 model configuration
PROMPT_VERSION="t5"
MODEL_VERSION="t5"
MODEL_PATH="downloads/t5-base"  # Using T5-base model

# Graph encoder configuration
GRAPH_TOWER="moleculestm"

# Training configuration
CHECKPOINT_FOLDER_PREFIX="_checkpoints/t5"
DATA_PATH="/root/autodl-tmp/TextChemT5-v0/omnimol_dataset/train"
REMARK="t5-base-full-finetune-15tasks-mixtrain-v2"

# Task configuration - all 15 tasks
TASK_CONFIG="forward:1.0/reagent:1.0/retrosynthesis:1.0/molcap:1.0/homolumo:1.0/solvent:1.0/catalyst:1.0/yield:1.0/experiment:1.0/tpsa:1.0/weight:1.0/dqa:1.0/logp:1.0/iupac:1.0/textguidedmolgen:1.0/molediting:1.0"

# DeepSpeed configuration
deepspeed train.py \
    --deepspeed scripts/zero_configs/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --base_model $MODEL_PATH \
    --language_backbone_name $MODEL_VERSION \
    --training_recipe t5 \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --graph_tower $GRAPH_TOWER \
    --mm_projector_type naive_linear \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/$REMARK \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 6e-4 \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --lr_scheduler_type "constant" \
    --logging_steps 50 \
    --bf16 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --moe_enable False \
    --task_config $TASK_CONFIG \
    --add_selfies True \
    --split_eval False \
    --val_ratio 0.1 \
    --logging_dir /root/tf-logs/$REMARK \
    --use_task_loss False
