from model_factory import load_lora_model, load_moe_lora_model,load_moe_lora_model_sequential,load_partial_model,load_pure_text_model
from transformers import HfArgumentParser, GenerationConfig, PreTrainedTokenizer, set_seed
from dataclasses import dataclass, field
from pathlib import Path
import time
import torch
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
from data_pipe import conversation_lib
from torch.utils.data import DataLoader
from model.modeling_llava import GraphLlavaForConditionalGeneration
import json
from typing import Sequence, Dict, Tuple, List
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import os
from metric_factory import calc_fingerprints, calc_mocap_metrics, calc_mol_trans, compute_mae, calc_iupac_metrics
from accelerate import Accelerator
from accelerate.utils import gather_object, InitProcessGroupKwargs
from datetime import timedelta
from metric_factory import calc_exp_metrics, compute_extracted_mae, compute_extracted_SCF_mae, compute_r2
import torch.distributed as dist
from loggers import WrappedLogger

logger = WrappedLogger(__name__)

local_rank = os.environ.get("LOCAL_RANK", -1)
if int(local_rank) != -1:
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
    accelerator = Accelerator(kwargs_handlers=[kwargs])

IGNORE_INDEX = -100


def apply_chat_template(message, tokenizer, has_image):
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)
    
    prompt = conv.get_prompt()
    
    return prompt


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

@dataclass
class EvalArguments:
    model_type: str = field(default="lora")
    task: str = field(default="fwd_pred")
    model_path: str = field(default=None)
    metric_path: str = field(default=None)
    language_backbone: str = field(default="checkpoints/phi3-mini")
    prompt_version: str = field(default="phi3")
    graph_tower: str = field(default="himol")
    graph_path: str = field(default=None)
    num_beams: int = field(default=1)
    top_p: float = field(default=1.0)
    temperature: float = field(default=0.2)
    data_path: str = field(default="forward_reaction_prediction.json")
    output_path: str = field(default="eval_result")
    batch_size: int = field(default=1)
    dtype: str = field(default="bfloat16", metadata={"choices": ["bfloat16", "float16", "float"]})
    use_flash_atten:bool = field(default=True)
    device:str = field(default="cuda", metadata={"choices": ["cpu", "cuda"]})
    add_selfies: bool = field(default=True)
    is_training: bool = False
    max_new_tokens: int = field(default=512)
    repetition_penalty: float = field(default=1.0)
    task_embed: str = field(default=None)
    task_definition: bool = field(default=False)
    task_identifier: bool = field(default=False)
    inference_on_train: bool = field(default=False)
    local_rank: int=field(default=-1)
    if_smiles: bool = field(default=False)
    seed: int = field(default=42)
    
    
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

def distributed_eval(loader, model, generation_config, tokenizer):
    all_batches = []
    for data in loader:
        all_batches.append(data)
    accelerator.wait_for_everyone()
    output = []
    cnt = 0  
    with accelerator.split_between_processes(all_batches) as batch:
        pbar = tqdm(total=len(batch), desc=f"[rank{local_rank}]")
        for each in batch:
            input_ids = each["input_ids"].to(args.device)
            graphs = each["graphs"]
            if isinstance(graphs, Batch):
                graphs = graphs.to(args.device)
            else:
                graphs = [x.to(args.device) if x is not None else None for x in graphs]
            output_ids = model.generate(
                input_ids,
                graphs=graphs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                use_cache=True,
                attention_mask=each["attention_mask"].to(args.device),
                this_task_ids=each["this_task_ids"].to(args.device),
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

def vanilla_eval(loader, model, generation_config, tokenizer):
    output = []
    for idx, batch in enumerate(loader):

        if isinstance(batch["graphs"], Batch):
            graphs = batch["graphs"].to(args.device)
        else:
            graphs = [x.to(args.device) if x is not None else None for x in batch["graphs"]]
        input_ids = batch["input_ids"].to(args.device)
        output_ids = model.generate(
            input_ids,
            graphs=graphs,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            use_cache=True,
            attention_mask=batch["attention_mask"].to(args.device),
            this_task_ids=batch["this_task_ids"].to(args.device),
            generation_config=generation_config,
            pad_token_id=tokenizer.eos_token_id
        )
        
        for idx, (result, input_id, prompt, gt) in enumerate(zip(output_ids, input_ids, batch["prompt"], batch["gt"])):
            this_output = {
                "prompt": prompt,
                "gt": gt,
                "pred": tokenizer.decode(result[input_id.shape[0]:])
            }
            output.append(this_output)
            # print("\n", this_output, "\n")
    
    return output

@torch.inference_mode
def start_eval(args: EvalArguments):
    tokenizer, model = build_pretrained_model(args.model_type, args.model_path, args.language_backbone, args.graph_path, args.use_flash_atten, args.task_embed)
    tokenizer.padding_side="left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
        
    if args.use_flash_atten:
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
    
    dataset = TASK_MAP[args.task](
        data_path=args.data_path,
        tokenizer=tokenizer,
        add_selfies=args.add_selfies,
        for_test=True
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=GraphEvalCollator(tokenizer), drop_last=False)
    if int(local_rank) != -1:
        output = distributed_eval(loader, model, generation_config, tokenizer)
    else:
        output = vanilla_eval(loader, model, generation_config, tokenizer)
    
    return output, tokenizer




if __name__ == "__main__":
    parser = HfArgumentParser((EvalArguments))
    args = parser.parse_args_into_dataclasses()[0]
    set_seed(args.seed)
    assert args.batch_size == 1, "Batched evaluation is under development!"


    if os.path.exists(args.output_path):
        output, tokenizer = start_eval(args)
        if int(local_rank) in [-1, 0]:
            print('File exists !')
            metric_path = args.metric_path.split("/")[:-1]
            metric_path = "/".join(metric_path)
            if not os.path.exists(metric_path):
                Path(metric_path).mkdir(parents=True) 
            with open(args.metric_path, mode="w") as f:
                json.dump('Evaluation Results', f, indent=2)
                f.close()   
    else:
        output, tokenizer = start_eval(args)
        # print(output,'+++'*20)
        if int(local_rank) in [-1, 0]:
            path = args.output_path.split("/")[:-1]
            path = "/".join(path)
            file = args.output_path.split("/")[-1]
            if not os.path.exists(path):
                Path(path).mkdir(parents=True, exist_ok=True)
            
            metric_path = args.metric_path.split("/")[:-1]
            metric_path = "/".join(metric_path)
            if not os.path.exists(metric_path):
                Path(metric_path).mkdir(parents=True, exist_ok=True)

            with open(args.metric_path, mode="w") as f:
                json.dump('Evaluation Results', f, indent=2)
                f.close()
                
            with open(args.output_path, mode="w") as f:
                json.dump(output, f, indent=2)
                f.close()
    if dist.is_initialized():
        dist.barrier()
    time.sleep(5)

    EOS_MAP = {
        "phi": "<|endoftext|>",
        "phi3": "<|end|>",
        "llama3": "<|eot_id|>",
        "tinyllama": "</s>"
    }
        
    if int(local_rank) in [0, -1]:
        if args.task in ["forward", "reagent", "retrosynthesis", "solvent", "catalyst", "iupac","textguidedmolgen","molediting"]:
            calc_mol_trans(args.output_path, args.metric_path, EOS_MAP[args.prompt_version])
            calc_fingerprints(args.output_path, args.metric_path, eos_token=EOS_MAP[args.prompt_version])
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained(args.language_backbone)


        if args.task in ["molcap", "dqa"]:
            if tokenizer.pad_token is None and args.prompt_version == "llama3":
                tokenizer.pad_token = "<|finetune_right_pad_id|>"
            calc_mocap_metrics(args.output_path, args.metric_path, EOS_MAP[args.prompt_version], tokenizer)

        if args.task == "homolumo":
            compute_mae(args.output_path, args.metric_path, EOS_MAP[args.prompt_version])

        if args.task == "yield_SM":
            compute_r2(args.output_path, args.metric_path, EOS_MAP[args.prompt_version])

        if args.task == "yield_BH":
            compute_r2(args.output_path, args.metric_path, EOS_MAP[args.prompt_version])

        if args.task in ["logp", "weight", "tpsa", "complexity"]:
            compute_extracted_mae(args.output_path, args.metric_path, EOS_MAP[args.prompt_version])

        if args.task == "scf":
            compute_extracted_SCF_mae(args.output_path, args.metric_path, EOS_MAP[args.prompt_version])

        if args.task == "experiment":
            calc_exp_metrics(args.output_path, args.metric_path, EOS_MAP[args.prompt_version], tokenizer)
        
        # if args.task == "iupac":
        #     if tokenizer.pad_token is None and args.prompt_version == "llama3":
        #         tokenizer.pad_token = "<|finetune_right_pad_id|>"
        #     calc_iupac_metrics(args.output_path, args.metric_path, EOS_MAP[args.prompt_version], tokenizer)
            
    dist.barrier()
            

