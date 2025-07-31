

# Base remark for experiment identification
BASE_REMARK="textchemt5-test-v2"

# Model checkpoint path
MODEL_BASE_PATH="/root/autodl-tmp/TextChemT5-v0/_checkpoints/t5/t5-base-full-finetune-15tasks-mixtrain-v2/checkpoint-2640"

# ------------------------
# Fixed parameters
# ------------------------
TYPE="lora+moe"
PROMPT="llama3"
BACKBONE="OmniMol/checkpoints/Llama-3.2-1B-Instruct"
GRAPH_TOWER="moleculestm"
GRAPH_PATH="OmniMol/checkpoints/moleculestm.pth"
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

# ------------------------
# Task list for evaluation
# ------------------------
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
    "dqa"
    "scf"
    "logp"
    "weight"
    "tpsa"
    "complexity"
    "experiment"
    "iupac"
    "textguidedmolgen"
    "molediting"
)

# ------------------------
# T5-specific parameters (comment out if not using T5)
# ------------------------
# TYPE="t5"
# PROMPT="t5"
# BACKBONE="downloads/t5-base"
# MODEL_BASE_PATH="/path/to/your/t5/checkpoint"
# DTYPE="bfloat16"
# USE_FLASH_ATTEN=False  # T5 doesn't support flash attention

# ------------------------
# Function to determine if flash attention should be used
# ------------------------
get_flash_atten_flag() {
    if [ "$TYPE" == "t5" ]; then
        echo "False"
    else
        echo "True"
    fi
}

# Get the appropriate flash attention setting
USE_FLASH_ATTEN=$(get_flash_atten_flag)

# ------------------------
# Loop through all tasks
# ------------------------
for TASK in "${TASK_LIST[@]}"; do
    # Set data path based on task
    case "$TASK" in
        "forward")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/forward_reaction_prediction.json"
            ;;
        "reagent")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/reagent_prediction.json"
            ;;
        "retrosynthesis")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/retrosynthesis.json"
            ;;
        "homolumo")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/property_prediction.json"
            ;;
        "molcap")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/molcap_test.json"
            ;;
        "solvent")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/solvent_pred.json"
            ;;
        "catalyst")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/catalyst_pred.json"
            ;;
        "yield_BH")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/yields_regression_BH.json"
            ;;
        "yield_SM")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/yields_regression_SM.json"
            ;;
        "dqa")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/3d_moit_no_homolumo_filtered_test.json"
            ;;
        "scf")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/3d_moit_no_homolumo_filtered_test.json"
            ;;
        "logp")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/3d_moit_no_homolumo_filtered_test.json"
            ;;
        "weight")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/3d_moit_no_homolumo_filtered_test.json"
            ;;
        "tpsa")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/3d_moit_no_homolumo_filtered_test.json"
            ;;
        "complexity")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/3d_moit_no_homolumo_filtered_test.json"
            ;;
        "experiment")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/exp_procedure_pred.json"
            ;;
        "iupac")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/iupac2selfies.json"
            ;;
        "textguidedmolgen")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/text_guided_molgen.json"
            ;;
        "molediting")
            DATA_PATH="OmniMol/Molecule-oriented_Instructions/evaluate/molecule_editing.json"
            ;;
        *)
            echo "Warning: Unknown TASK: $TASK"
            continue
            ;;
    esac

    # Generate REMARK with task name
    REMARK="${BASE_REMARK}-${TASK}"
    
    # Handle special case for yield regression tasks
    if [ "$TASK" == "yield_BH" ] || [ "$TASK" == "yield_SM" ]; then
        MODEL_REMARK="${BASE_REMARK}-yields_regression"
    else
        MODEL_REMARK=$REMARK
    fi

    # Output file paths
    OUTPUT_PATH="OmniMol/eval_result/save_all_tasks/${BASE_REMARK}/${TASK}-${TYPE}-${PROMPT}-answer.json"
    METRIC_PATH="OmniMol/eval_result/save_all_tasks_metric/${BASE_REMARK}/${TASK}-${TYPE}-${PROMPT}-metric.json"
    
    # Model path
    MODEL_PATH="${MODEL_BASE_PATH}"

    # Create output directories if they don't exist
    mkdir -p "$(dirname "$OUTPUT_PATH")"
    mkdir -p "$(dirname "$METRIC_PATH")"

    # Print task information
    echo "--------------------------------------"
    echo " Running TASK:         $TASK"
    echo " Model type:           $TYPE"
    echo " Data file:            $DATA_PATH"
    echo " Model path:           $MODEL_PATH"
    echo " Metric path:          $METRIC_PATH"
    echo " Output file:          $OUTPUT_PATH"
    echo " Flash attention:      $USE_FLASH_ATTEN"
    echo "--------------------------------------"

    # Execute evaluation command
    # Use accelerate for distributed evaluation if needed
    if [ "$TYPE" == "t5" ]; then
        # For T5 models, you might want to use single GPU due to memory constraints
        python eval_engine.py \
            --model_type "$TYPE" \
            --task "$TASK" \
            --model_path "$MODEL_PATH" \
            --metric_path "$METRIC_PATH" \
            --language_backbone "$BACKBONE" \
            --prompt_version "$PROMPT" \
            --graph_tower "$GRAPH_TOWER" \
            --graph_path "$GRAPH_PATH" \
            --num_beams "$NUM_BEAMS" \
            --top_p "$TOP_P" \
            --temperature "$TEMPERATURE" \
            --data_path "$DATA_PATH" \
            --output_path "$OUTPUT_PATH" \
            --batch_size "$BATCH_SIZE" \
            --dtype "$DTYPE" \
            --use_flash_atten "$USE_FLASH_ATTEN" \
            --device "$DEVICE" \
            --add_selfies "$ADD_SELFIES" \
            --is_training "$IS_TRAINING" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --repetition_penalty "$REPETITION_PENALTY"
    else
        # For other models, use accelerate for distributed evaluation
        /root/anaconda3/bin/accelerate launch eval_engine.py \
            --model_type "$TYPE" \
            --task "$TASK" \
            --model_path "$MODEL_PATH" \
            --metric_path "$METRIC_PATH" \
            --language_backbone "$BACKBONE" \
            --prompt_version "$PROMPT" \
            --graph_tower "$GRAPH_TOWER" \
            --graph_path "$GRAPH_PATH" \
            --num_beams "$NUM_BEAMS" \
            --top_p "$TOP_P" \
            --temperature "$TEMPERATURE" \
            --data_path "$DATA_PATH" \
            --output_path "$OUTPUT_PATH" \
            --batch_size "$BATCH_SIZE" \
            --dtype "$DTYPE" \
            --use_flash_atten "$USE_FLASH_ATTEN" \
            --device "$DEVICE" \
            --add_selfies "$ADD_SELFIES" \
            --is_training "$IS_TRAINING" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --repetition_penalty "$REPETITION_PENALTY"
    fi

    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "✅ Successfully completed TASK: $TASK"
    else
        echo "❌ Failed to complete TASK: $TASK"
    fi
    
    echo ""
done

echo "============================================"
echo "All tasks completed!"
echo "Results saved to: OmniMol/eval_result/save_all_tasks/${BASE_REMARK}/"
echo "Metrics saved to: OmniMol/eval_result/save_all_tasks_metric/${BASE_REMARK}/"
echo "============================================"