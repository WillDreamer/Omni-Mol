import os
from transformers import PreTrainedModel
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from transformers.generation.configuration_utils import GenerationConfig
from transformers import PreTrainedTokenizer
from loggers import WrappedLogger
from accelerate.utils import gather_object
import torch
from model_factory import load_lora_model, load_moe_lora_model, load_moe_lora_model_sequential, load_partial_model, load_pure_text_model
from metric_factory import calc_fingerprints, calc_mocap_metrics, calc_mol_trans, compute_mae, calc_iupac_metrics, calc_exp_metrics, compute_extracted_mae, compute_extracted_SCF_mae, compute_r2
from model.modeling_llava import GraphLlavaForConditionalGeneration
from data_pipe import conversation_lib
from data_pipe.datasets import (
    ForwardPredDataset, 
    RetrosynDataset, 
    ReagentPredDataset, 
    MolcapDataset,
    PropertyPredDataset,
    SolventPredDataset,
    CatalystPredDataset,
    YieldRegressionDataset,
    ExpProcedurePrediction,
    SCFPrediction,
    ComlexityPrediction,
    TPSAPrediction,
    WeightPrediction,
    DescriptionQA,
    LogPPrediction,
    IUPAC,
    TextGuidedMolGen,
    MolEditing
)
from typing import Sequence, Dict, Tuple, List
from dataclasses import dataclass, field
from helpers import save_json
import wandb
from accelerate.utils import gather_object, InitProcessGroupKwargs

from datetime import timedelta 

local_rank = os.environ.get("LOCAL_RANK", -1)
if int(local_rank) != -1:
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
logger = WrappedLogger(__name__)

@dataclass       
class GraphEvalCollator(object):
    """Collate graph-QA examples for supervised fine-tuning."""
    
    tokenizer: PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, gt, prompt, task_id = self._extract_tensors(instances, ("input_ids", "gt", "prompt", "this_task_ids"))
        
        batch_input = self._pad_sequence(input_ids, self.tokenizer.pad_token_id)
        task_ids = torch.cat(task_id, dim=0)
        batch = {
            'input_ids': batch_input["input_ids"][:, :self.tokenizer.model_max_length],
            'gt': gt,
            'prompt': prompt,
            'this_task_ids': task_ids,
            'attention_mask': batch_input["attention_mask"][:, :self.tokenizer.model_max_length],
        }
        if 'graphs' in instances[0]:
            graph_batch = []
            for instance in instances:
                if instance["graphs"] is not None:
                    graph_batch.append(self._convert_dict_to_Data(instance["graphs"]))
                else:
                    graph_batch.append(None)
            batch["graphs"] = graph_batch
            # batch['graphs'] = Batch.from_data_list(
            #     [self._convert_dict_to_Data(instance["graphs"]) for instance in instances]
            # )
        return batch
    
    def _extract_tensors(self, instances, keys: Tuple[str, str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return tuple([instance[key] for instance in instances] for key in keys)

    def _pad_sequence(self, sequence: List[torch.Tensor], padding_value: int) -> torch.Tensor:
        return self.tokenizer.pad({"input_ids": sequence}, padding=True)

    def _convert_dict_to_Data(self, data_dict: Dict) -> Data:
        if getattr(data_dict, "num_part", None) is not None: # which means we are using himol
            return Data(
            x=torch.asarray(data_dict.x),
            edge_index=torch.asarray(data_dict.edge_index),
            edge_attr=torch.asarray(data_dict.edge_attr),
            num_part=data_dict.num_part
            )
            
        return Data(
            x=torch.asarray(data_dict['node_feat']),
            edge_attr=torch.asarray(data_dict['edge_feat']),
            edge_index=torch.asarray(data_dict['edge_index']),
        )

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--metric_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--language_backbone", type=str, required=True)
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--use_flash_attn", type=bool, default=True)
    parser.add_argument("--task_embed", type=bool, default=False)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--add_selfies", type=bool, default=True)
    parser.add_argument("--prompt_version", type=str, default="llama3")
    parser.add_argument("--eval_all_epochs", type=bool, default=False)
    parser.add_argument("--top_p", type=float, default=1.0)

    args = parser.parse_args()
    
    return args


def get_all_ckpts(save_path: str, eval_all_epochs: bool):
    saved_ckpts = [
        d for d in os.listdir(save_path)
        if os.path.isdir(os.path.join(save_path, d))
    ]
    saved_ckpts.sort(key=lambda x:x.split("checkpoint-")[1])
    if not eval_all_epochs:
        saved_ckpts = [saved_ckpts[0]]
        
    return saved_ckpts

def evaluation_loop(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer,
    dataloader: DataLoader,
    device: str,
    temperature: float,
    top_p: float,
    num_beams: int,
    max_new_tokens: int,
    repetition_penalty: float,
    generation_config: GenerationConfig
):
    all_batches = []
    for data in dataloader:
        all_batches.append(data)
    accelerator.wait_for_everyone()
    output = []
    cnt = 0  
    with accelerator.split_between_processes(all_batches) as batch:
        pbar = tqdm(total=len(batch), desc=f"[rank{accelerator.local_process_index}]")
        for each in batch:
            input_ids = each["input_ids"].to(device)
            graphs = each["graphs"]
            if isinstance(graphs, Batch):
                graphs = graphs.to(device)
            else:
                graphs = [x.to(device) if x is not None else None for x in graphs]
            output_ids = model.generate(
                input_ids,
                graphs=graphs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                use_cache=True,
                attention_mask=each["attention_mask"].to(device),
                this_task_ids=each["this_task_ids"].to(device),
                generation_config=generation_config
            )
            
            for idx, (result, input_id, prompt, gt) in enumerate(zip(output_ids, input_ids, each["prompt"], each["gt"])):
                this_output = {
                    "prompt": prompt,
                    "gt": gt,
                    "pred": tokenizer.decode(result[input_id.shape[0]:])
                }
                output.append(this_output)
                # if cnt < 10:
                #     print("\n", this_output, "\n")
            pbar.update(1)
                
            cnt += 1
    logger.info("Gathering object from processes...")       
    output = gather_object(output)
    accelerator.wait_for_everyone()

    return output
    

MODEL_LOADER_MAP = {
    "lora": load_lora_model,
    "lora+moe": load_moe_lora_model,
    "sequential": load_moe_lora_model_sequential,
    "partial":load_partial_model,
    "puretext": load_pure_text_model
}

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float": torch.float
}


TASK_MAP = {
    "forward": ForwardPredDataset,
    "reagent": ReagentPredDataset,
    "retrosynthesis": RetrosynDataset,
    "molcap": MolcapDataset,
    "homolumo": PropertyPredDataset,
    "solvent": SolventPredDataset,
    "catalyst": CatalystPredDataset,
    "yield_BH": YieldRegressionDataset,
    "yield_SM": YieldRegressionDataset,
    "experiment": ExpProcedurePrediction,
    "scf": SCFPrediction,
    "complexity": ComlexityPrediction,
    "tpsa": TPSAPrediction,
    "weight": WeightPrediction,
    "dqa": DescriptionQA,
    "logp": LogPPrediction,
    "iupac": IUPAC,
    'textguidedmolgen': TextGuidedMolGen,
    "molediting": MolEditing
}

MAP2FILEMAME = {
    "forward": "forward",
    "reagent": "reagent",
    "retrosynthesis": "retrosynthesis",
    "molcap": "molcap_test",
    "homolumo": "property",
    "solvent": "solvent",
    "catalyst": "catalyst",
    "yield_BH": "yields_regression_BH",
    "yield_SM": "yields_regression_SM",
    "experiment": "exp_procedure_pred",
    "tpsa": "3d_moit",
    "weight": "3d_moit",
    "dqa": "3d_moit",
    "logp": "3d_moit",
    "iupac": "iupac2selfies",
    "textguidedmolgen": "text_guided",
    "molediting": "molecule_editing"
}

def build_pretrained_model(
    model_type,
    model_path,
    language_backbone,
    graph_path,
    use_flash_attn,
    task_embed_path=None
    ) -> tuple[PreTrainedTokenizer, GraphLlavaForConditionalGeneration]:
    
    tokenizer, model = MODEL_LOADER_MAP[model_type](model_path, language_backbone, graph_path, use_flash_attn, task_embed_path)
    return tokenizer, model    


def calc_metrics(tokenizer, task, output_path, metric_path, prompt_version):
    EOS_MAP = {
        "phi": "<|endoftext|>",
        "phi3": "<|end|>",
        "llama3": "<|eot_id|>",
        "tinyllama": "</s>"
    }
        
    if task in ["forward", "reagent", "retrosynthesis", "solvent", "catalyst", "iupac","textguidedmolgen","molediting"]:
        result = calc_mol_trans(output_path, metric_path, EOS_MAP[prompt_version])
        result += calc_fingerprints(output_path, metric_path, eos_token=EOS_MAP[prompt_version])
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(args.language_backbone)


    if task in ["molcap", "dqa"]:
        if tokenizer.pad_token is None and prompt_version == "llama3":
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        result = calc_mocap_metrics(output_path, metric_path, EOS_MAP[prompt_version], tokenizer)

    if task == "homolumo":
        result = compute_mae(output_path, metric_path, EOS_MAP[prompt_version])

    if task == "yield_SM":
        result = compute_r2(output_path, metric_path, EOS_MAP[prompt_version])

    if task == "yield_BH":
        result = compute_r2(output_path, metric_path, EOS_MAP[prompt_version])

    if task in ["logp", "weight", "tpsa", "complexity"]:
        result = compute_extracted_mae(output_path, metric_path, EOS_MAP[prompt_version])

    if task == "scf":
        result = compute_extracted_SCF_mae(output_path, metric_path, EOS_MAP[prompt_version])

    if task == "experiment":
        result = calc_exp_metrics(output_path, metric_path, EOS_MAP[prompt_version], tokenizer)
        
    return result

    
@torch.inference_mode
def start_eval(args):
    if accelerator.is_main_process:
        wandb.init(name=f"{args.model_path}", project="Omni-Mol-eval")
    all_ckpts = get_all_ckpts(args.model_path, args.eval_all_epochs)
    for ckpt in all_ckpts:
        logger.info("****************************")
        logger.info(f"Evaluation on {ckpt}")
        logger.info("****************************")    
        tokenizer, model = build_pretrained_model(args.model_type, os.path.join(args.model_path, ckpt), args.language_backbone, args.graph_path, args.use_flash_attn, args.task_embed)
        tokenizer.padding_side="left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token=tokenizer.eos_token
            
        if args.use_flash_attn:
            print("Using flash attention, will force the computation dtype to bfloat16...")
            assert args.device == "cuda", "Flash attention only supports running on CUDA devices!"
            model.to(torch.bfloat16)
        else:
            model.to(DTYPE_MAP[args.dtype])
            
        model.to(args.device)
        
        # print(model)
        
        generation_config = GenerationConfig.from_pretrained(args.language_backbone)
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.prompt_version]
        # print("Using conversation template of", args.prompt_version)
        # print("Conversation template:", conversation_lib.default_conversation)
        dataset_files = os.listdir(args.data_path)
        for task_name in TASK_MAP.keys():
            try:
                file_mask = [MAP2FILEMAME[task_name] in file_name for file_name in dataset_files]
                position = file_mask.index(True)
            except Exception as e:
                raise ValueError(f"Cannot locate file for {task_name} due to {e}")
            this_data_path = os.path.join(args.data_path, dataset_files[position])
            dataset = TASK_MAP[task_name](
                data_path=this_data_path,
                tokenizer=tokenizer,
                add_selfies=args.add_selfies,
                for_test=True
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=GraphEvalCollator(tokenizer), drop_last=False)
            
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            file_path = os.path.join(args.save_path, f"{task_name}_results.json")
            if os.path.exists(file_path):
                logger.info(f"Results for {task_name} already exists, skipping evaluation...")
            else:
                output = evaluation_loop(
                    model,
                    tokenizer,
                    loader,
                    args.device,
                    args.temperature,
                    args.top_p,
                    args.num_beams,
                    args.max_new_tokens,
                    args.repetition_penalty,
                    generation_config
                )
                save_json(output, file_path)
            if accelerator.is_main_process:
                
                result = calc_metrics(
                    tokenizer, 
                    task_name,
                    file_path,
                    args.metric_path,
                    args.prompt_version
                )
                for metric in result:
                    if accelerator.is_main_process:
                        wandb.log({f"{task_name}/{k}": v for k, v in metric.items()}, step=int(ckpt.split("checkpoint-")[1]))    
                        
                            
            accelerator.wait_for_everyone()
    
    
if __name__ == "__main__":
    args = parse_args()
    start_eval(args)
