# [Unifying Molecular and Textual Representations via Multi-task Language Modelling](https://proceedings.mlr.press/v202/christofidellis23a.html)

Reproducing "Unifying Molecular and Textual Representations via Multi-task Language Modelling"

## Reproducibility Overview
This directory provides a reproducibility guide for TextChemT5, including its hyperparameter configurations, training process, and training logs. TextChemT5 is a unified molecular language model based on the T5 architecture. It utilizes a pretrained T5 model (google-t5/t5-base from Hugging Face) to model the textual information of molecules and is trained in a unified framework across multiple molecular tasks. This guide follows the methodology of the original paper to execute all tasks proposed in Omni-Mol. Given the differences between the tasks in the two papers, we have fine-tuned TextChemT5 to maximize its potential and achieve optimal performance. For example, we increased the maximum number of tokens the model can process to enable more effective loss computation.


## Dataset
### Task list
- "forward"
- "reagent"
- "retrosynthesis"
- "homolumo"
- "molcap"
- "solvent"
- "catalyst"
- "yield_BH"
- "yield_SM"
- "dqa"
- "logp"
- "weight"
- "tpsa"
- "experiment"
- "moledit"
- "iupac2selfies"
- "moldesign"

The default setting for `--task_config' is:

```bash
"forward:1/retrosynthesis:1/reagent:1/homolumo:1/molcap:1/solvent:1/catalyst:1/yield:1/experiment:1/tpsa:1/weight:1/dqa:1/logp:1/iupac:1/textguidedmolgen:1/molediting:1"
```

## Train
### Finetune for all tasks
```bash
bash scripts/lora_finetunes/t5-base/mixtrain-t5.sh
```
As described in the paper, we use the same architecture as google-t5/t5-base. For further details on model, dataset, and training parameters and their explanations, please refer to `args.py` and the shell scripts mentioned above.

## Training logs
To provide greater transparency into our training process, we have made the log files for the learning rate, gradient norm, loss value, and epochs available at `/logs`. Furthermore, for a more transparent view of the model's parameters and activations, we provide a model profile, which can be found at `/model/model_profile.json`.

## Evaluation
If you have multiple GPUs, we support distributed inference
```bash
bash scripts/dist_eval_all.sh
```
For specific parameters and explanations regarding the inference stage, please refer to `eval_engine.py` and the shell scripts mentioned above.
