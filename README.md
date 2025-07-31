# Omni-Mol: Multitask Molecular Model for Any-to-any Modalities


## Release

[2025/07/28] 🔥 We release our first version of code

## 🌍 Environment Setup
1. Clone the repository and `cd` to the folder

```bash
git clone https://github.com/xxxxxxxx/OmniMol.git

cd OmniMol
```
2. (Optional) Environment Settings:

```bash
# Download the Miniconda installer script
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# Install to $HOME/miniconda3 in batch mode
bash ~/miniconda.sh -b -p $HOME/miniconda3

# Activate conda (only in the current shell)
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Accept the ToS for the main channel:
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# (Optional) Add conda to your default shell startup
conda init

# Reload shell config
source ~/.bashrc
```
or
```bash
bash setup_conda.sh
```

3. Install package of OmniMol through 
```bash
bash setup_omnimol.sh
```

4. If you download this repo from Anonymous GitHub, you may lack git submodule `peft`, you can add it by

```bash
git submodule add https://github.com/WillDreamer/peft.git peft
git submodule init
git submodule update
```

## 🚀 Weights
We provide the checkpoints of Omni-Mol reported in main table, you can find all of the files in the anonymous Huggingface repo: [https://huggingface.co/datasets/Omni-Mol/omnimol-ckpt.](https://huggingface.co/Omni-Mol/omnimol-ckpt)

All of our default setting can be found in `checkpoint-197148/config.json`.

## 📊 Dataset

### Data Download
We have also provided compressed packages of all our training and testing data, you can find all of the files in the anonymous Huggingface repo: [https://huggingface.co/datasets/Omni-Mol/omnimol-data.](https://huggingface.co/datasets/Omni-Mol/omnimol-data)

### Task list
- "forward"
- "reagent"
- "retrosynthesis"
-  "homolumo"
- "molcap"
- "solvent"
- "catalyst"
- "yield_BH"
- "yield_SM"
- "dqa"
- "scf"
- "logp"
- "weight"
- "tpsa"
- "complexity"
- "experiment"

The default setting for `--task_config' is:

```bash
"forward:1/retrosynthesis:1/reagent:1/homolumo:1/molcap:1/solvent:1/catalyst:1/yield:1/experiment:1/tpsa:1/weight:1/dqa:1/logp:1/iupac:1/textguidedmolgen:1/molediting:1"
```


## 🔥 Train
### Stage 1: Pre-training of Multi-modal Projector (Optional) 

```bash
bash scripts/pretrain.sh
```
or you can just utilize our provided `mm_projector.bin' which is automatically downloaded if you run `setup_omnimol.sh`.

### Stage 2: MoE + PEFT 

To follow the default setting, please run the code with:
```bash 
bash scripts/mixtrain_auto_eval.sh
```

Actually, we support multiple kinds of training recipe in `model_factory.py':

```bash 
MODEL_STAGE_MAP = {
    "lora": create_lora_model,
    "loramoe": create_lora_moe_model, 
    "sequential": load_moe_lora_model_sequential,
    "puretext": create_puer_text_model 
}

```
1️⃣ "lora" represents the pure lora mode without MoE expansion

2️⃣ "loramoe" represents our design of MoE + PEFT

3️⃣ "sequential" represents the continual pre-training mode instead of our unified SFT

4️⃣ "puretext" represents the abltion of merging Graph modality into text prompt


### Explanation of Training Script Arguments

| Argument | Description |
|----------|-------------|
| `--training_recipe` | Specifies the training recipe discussed above. |
| `--use_alpha` | Enables dynamical scaling used in LoRA. |
| `--task_config` | Define the task list. |
| `--model_name_or_path` | Base pretrained language model (e.g., LLaMA 3.2-1B). |
| `--base_model` | Base model used for initialization (same as `model_name_or_path`). |
| `--language_backbone_name` | Name or version of the language backbone used in multimodal setup. |
| `--version` | Prompt format versioning. |
| `--data_path` | Path to training data. |
| `--data_type` | Task name or type, used to select data preprocessing logic. |
| `--graph_tower` | Name of the graph model component (e.g., GNN). |
| `--mm_projector_type` | Type of multimodal projection layer. |
| `--graph_init_checkpoint` | Checkpoint path to initialize the GNN. |
| `--pretrain_mm_mlp_adapter` | Path to a pretrained MLP adapter for multimodal fusion. |
| `--bf16 True` | Enables bfloat16 precision. |
| `--output_dir` | Output directory for checkpoints and logs. |
| `--num_train_epochs` | Number of training epochs. |
| `--per_device_train_batch_size` | Batch size per GPU for training. |
| `--per_device_eval_batch_size` | Batch size per GPU for evaluation. |
| `--gradient_accumulation_steps` | Number of steps to accumulate gradients. |
| `--stop_epoch` | Training will be stopped early at this epoch. |
| `--eval_strategy` | Evaluation strategy. |
| `--eval_steps` | Evaluation interval (not used if eval is disabled). |
| `--split_eval` | Whether to split evaluation by data subsets. |
| `--val_ratio` | Ratio of data used for validation. |
| `--eval_on_start` | If true, performs evaluation before training begins. |
| `--save_strategy` | Saves model checkpoint at the end of each epoch. |
| `--save_total_limit` | Maximum number of checkpoints to retain. |
| `--learning_rate` | Initial learning rate. |
| `--weight_decay` | No weight decay (regularization). |
| `--warmup_ratio` | Warm-up proportion for learning rate scheduler. |
| `--lr_scheduler_type` | Scheduler type; cosine annealing in this case. |
| `--logging_steps` | Interval for logging metrics. |
| `--tf32` | Enables TensorFloat-32 on compatible hardware. |
| `--model_max_length` | Maximum input sequence length. |
| `--gradient_checkpointing` | Saves memory by checkpointing activations. |
| `--dataloader_num_workers` | Number of workers for data loading. |
| `--report_to` | Logging target (e.g., TensorBoard, WandB). |
| `--logging_dir ` | Directory where training logs are saved. |
| `--moe_class` | MoE routing class implementation (e.g., from DeepSeek). |
| `--moe_mode` | Controls which layer(s) apply MoE. |
| `--ep_size` | Expert parallelism size (used in MoE). |
| `--num_experts` | Total number of experts in the MoE layer. |
| `--use_residual` | Enables residual connections in expert mixing. |
| `--router_aux_loss_coef` | Coefficient for router auxiliary loss in MoE. |
| `--is_training` | Whether the model is in training mode. |
| `--top_k_experts` | Number of experts selected per token. |
| `--use_task_loss` | Whether to include task-specific loss term. |
| `--ddp_find_unused_parameters` | Disables search for unused parameters in DDP (for efficiency). |
| `--norm_topk_prob` | Whether to normalize top-k routing probabilities. |

More details can be found in `args.py'.

## 🌟 Evaluation

We support the auto evaluation after training in 

```bash 
bash scripts/mixtrain_auto_eval.sh
```

We also support separate evaluation with distributed inference

```bash
bash scripts/dist_eval_all_epoch.sh
```

For example, after downloading `checkpoint-197148` in [https://huggingface.co/Omni-Mol/omnimol-ckpt](https://huggingface.co/Omni-Mol/omnimol-ckpt), please set the correct path to the evaluation data on your system. You can then run the following commands directly to reproduce our results. Here, `197148` refers to the training step at which the checkpoint was saved.

```bash
bash scripts/dist_eval_all_epoch.sh
```

Please claim the task for evaluation in `TASK_MAP', and the evaluation mode in `MODEL_LOADER_MAP' with `--model_type' in scripts.

## 😈 Text-Chem T5 Re-implementaion

All of our reproduction details are provided under the `TextChemT5-reimplement` directory (i.e., `TextChemT5-v0-main`), including log files (`TextChemT5-reimplement/logs`) and model profiles (containing activation and model parameter information in `TextChemT5-reimplement/model/model_profile.json`). For more information, please refer to the `TextChemT5-reimplement/README.md`.





