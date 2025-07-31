#!/usr/bin/env bash


TYPE="loramoe"
PROMPT="llama3"
BACKBONE="meta-llama/Llama-3.2-1B-Instruct"
GRAPH_TOWER="moleculestm"
GRAPH_PATH="assets/moleculestm.pth"
BATCH_SIZE=1
DTYPE="bfloat16"
DEVICE="cuda"
MAX_NEW_TOKENS=512
NUM_BEAMS=1
TOP_P=1.0
TEMPERATURE=0.2
REPETITION_PENALTY=1.0
ADD_SELFIES=True
IS_TRAINING=False
SEED=42


EPOCH_LIST=(14)
CHECKPOINT_LIST=(197148)


TASK_LIST=(
     "forward"
     "reagent"
     "retrosynthesis"
     "homolumo"
     "molcap"
     "solvent"
     "catalyst"
     "yield_BH"
     "yield_SM"
    "logp"
    "weight"
    "tpsa"
    "experiment"
    "iupac2selfies"
    "molediting"
    "textguidedmolgen"
)

for i in "${!EPOCH_LIST[@]}"; do
    EPOCH=${EPOCH_LIST[$i]}
    CKPT=${CHECKPOINT_LIST[$i]}

    BASE_REMARK="llama-1B-deepseek-moe-5EP-qurater-sharedEP-clip-alpha-embed-Tok2-16tasks-seed${SEED}-temp${TEMPERATURE}-Epoch${EPOCH}"
    MODEL_BASE_PATH="_checkpoints/checkpoint-${CKPT}"

    for TASK in "${TASK_LIST[@]}"; do
        case "$TASK" in
            "forward") DATA_PATH="data/evaluate/forward_reaction_prediction.json" ;;
            "reagent") DATA_PATH="data/evaluate/reagent_prediction.json" ;;
            "retrosynthesis") DATA_PATH="data/evaluate/retrosynthesis.json" ;;
            "homolumo") DATA_PATH="data/evaluate/property_prediction.json" ;;
            "molcap") DATA_PATH="data/evaluate/molcap_test.json" ;;
            "solvent") DATA_PATH="data/evaluate/solvent_pred.json" ;;
            "catalyst") DATA_PATH="data/evaluate/catalyst_pred.json" ;;
            "yield_BH") DATA_PATH="data/evaluate/yields_regression_BH.json" ;;
            "yield_SM") DATA_PATH="data/evaluate/yields_regression_SM.json" ;;
            "iupac") DATA_PATH="data/evaluate/iupac2selfies.json" ;;
            "molediting") DATA_PATH="data/evaluate/molecule_editing.json" ;;
            "textguidedmolgen") DATA_PATH="data/evaluate/text_guided_mol_generation.json" ;;
            "dqa"|"scf"|"logp"|"weight"|"tpsa"|"complexity")
                DATA_PATH="data/evaluate/3d_moit_no_homolumo_filtered_test.json"
                ;;
            "experiment") DATA_PATH="dta/evaluate/exp_procedure_pred.json" ;;
            *) echo "Warning: Unknown TASK: $TASK"; continue ;;
        esac

        REMARK="${BASE_REMARK}-${TASK}"
        if [[ "$TASK" == "yield_BH" || "$TASK" == "yield_SM" ]]; then
            MODEL_REMARK="${BASE_REMARK}-yields_regression"
        else
            MODEL_REMARK=$REMARK
        fi

        PPP_PATH="eval_result/save_all_tasks/${BASE_REMARK}/${TASK}-${TYPE}-${PROMPT}-answer.json"
        METRIC_PATH="eval_result/save_all_tasks_metric/${BASE_REMARK}/${TASK}-${TYPE}-${PROMPT}-metric.json"

        echo "--------------------------------------"
        echo "Epoch:                $EPOCH"
        echo "Checkpoint:           $CKPT"
        echo "Running TASK:         $TASK"
        echo "Data file:            $DATA_PATH"
        echo "Model path:           $MODEL_BASE_PATH"
        echo "Metric path:          $METRIC_PATH"
        echo "Output file:          $PPP_PATH"
        echo "--------------------------------------"

        accelerate launch --main_process_port 29519 eval_engine.py \
            --model_type "$TYPE" \
            --task "$TASK" \
            --model_path "$MODEL_BASE_PATH" \
            --metric_path "$METRIC_PATH" \
            --language_backbone "$BACKBONE" \
            --prompt_version "$PROMPT" \
            --graph_tower "$GRAPH_TOWER" \
            --graph_path "$GRAPH_PATH" \
            --num_beams "$NUM_BEAMS" \
            --top_p "$TOP_P" \
            --temperature "$TEMPERATURE" \
            --data_path "$DATA_PATH" \
            --output_path "$PPP_PATH" \
            --batch_size "$BATCH_SIZE" \
            --dtype "$DTYPE" \
            --use_flash_atten True \
            --device "$DEVICE" \
            --add_selfies "$ADD_SELFIES" \
            --is_training "$IS_TRAINING" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --repetition_penalty "$REPETITION_PENALTY" \
            --seed "$SEED"

        echo "Finish TASK: $TASK"
        echo
    done
done

echo "All epochs and tasks finished!"
