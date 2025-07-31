import os
from transformers import PreTrainedModel
from args import TrainingArguments
import json
from pathlib import Path
import torch
from torch import nn
from contextlib import contextmanager
from transformers.modeling_utils import _init_weights
from loggers import WrappedLogger

logger = WrappedLogger(__name__)


class NestedTensor(object):
    def __init__(self, tensors: torch.Tensor, mask: torch.Tensor):
        self.tensors: torch.Tensor = tensors
        self.mask: torch.Tensor = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask
    
    @property
    def dtype(self):
        return self.tensors.dtype
    
    @property
    def shape(self):
        return self.tensors.shape

    def __repr__(self):
        return str(self.tensors)


def initialize_distributed_training(local_rank):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        if int(local_rank) == 0:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
    

def model_profiler(model: PreTrainedModel, save_dir: str):
    param = model.num_parameters()
    model_structure = str(model).split("\n")
    activated_params = [name for name, param in model.named_parameters() if param.requires_grad]
            
    logger.info(f"Model Num Param: {param}", on_rank0=True)
    logger.info(f"Model Structure...", on_rank0=True)
    logger.info(model, on_rank0=True)
    logger.info("Activated Parameters...", on_rank0=True)
    logger.info("\n".join(activated_params), on_rank0=True)
            
    output_dir = Path(save_dir)
    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True)
        
    profile = {
        "Num Param": param,
        "Model structure": model_structure,
        "Activated paramteres": activated_params
    }
    with open(output_dir/"model_profile.json", mode="w") as f:
        json.dump(profile, f, indent=4)
            

TORCH_INIT_FUNCTIONS = {
    "uniform_": nn.init.uniform_,
    "normal_": nn.init.normal_,
    "trunc_normal_": nn.init.trunc_normal_,
    "constant_": nn.init.constant_,
    "xavier_uniform_": nn.init.xavier_uniform_,
    "xavier_normal_": nn.init.xavier_normal_,
    "kaiming_uniform_": nn.init.kaiming_uniform_,
    "kaiming_normal_": nn.init.kaiming_normal_,
    "uniform": nn.init.uniform,
    "normal": nn.init.normal,
    "xavier_uniform": nn.init.xavier_uniform,
    "xavier_normal": nn.init.xavier_normal,
    "kaiming_uniform": nn.init.kaiming_uniform,
    "kaiming_normal": nn.init.kaiming_normal,
}


@contextmanager
def no_init_weights(_enable=True):
    """
    Context manager to globally disable weight initialization to speed up loading large models.

    TODO(Patrick): Delete safety argument `_enable=True` at next major version. .
    """
    global _init_weights
    old_init_weights = _init_weights

    if _enable:
        _init_weights = False

        def _skip_init(*args, **kwargs):
            pass

        # # Save the original initialization functions
        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, _skip_init)
    try:
        yield
    finally:
        _init_weights = old_init_weights
        if _enable:
            # # Restore the original initialization functions
            for name, init_func in TORCH_INIT_FUNCTIONS.items():
                setattr(torch.nn.init, name, init_func)
                
                
def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
    
    
def maybe_zero_3(param: torch.Tensor, ignore_status: bool=False, name: str=None) -> torch.Tensor:
    """Safely gather zero 3 params

    Args:
        param (torch.Tensor): parameters to be gathered
        ignore_status (bool, optional): Whether to ignore not available status. Defaults to False.
        name (str, optional): name of the parameter. Defaults to None.

    Returns:
        torch.Tensor: Parameters recovered from zero status 3
    """
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param
    
    
# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for bias_name, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return

    
def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def seperate_save_partial(
    args, 
    checkpoint_dir: str, 
    model: nn.Module
    ):
    """Save partial trainables weights seperately
    
    Args:
        args (DataClass): training args used in huggingface training
        checkpoint_dir (str): dir used to save
        model (nn.Module): model to be saved
    """
    trainable_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters())

    if args.local_rank in [-1, 0]:  # On master rank
        # save things
        torch.save(trainable_state_dict, os.path.join(checkpoint_dir, 'trainables.bin'))
    
def seperate_save_lora(
    args, 
    checkpoint_dir: str, 
    model: nn.Module
    ):
    """Save LoRA weights and non-lora trainables weights seperately
    
    Args:
        args (DataClass): training args used in huggingface training
        checkpoint_dir (str): dir used to save
        model (nn.Module): model to be saved
    """
    if getattr(args, "lora_enable", False):  # if lora enabled
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), args.lora_bias
        )  # Find all LoRA weights and biases, if it's in ZeRO-3, this function will handle it.
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )  # get model state dict not containing LoRA
        if args.local_rank in [-1, 0]:  # On master rank
            # save things
            model.config.save_pretrained(checkpoint_dir)  # save config
            model.save_pretrained(checkpoint_dir, state_dict=state_dict)  # save LoRA state_dict
            torch.save(non_lora_state_dict, os.path.join(checkpoint_dir, 'non_lora_trainables.bin'))  # save non-lora
            
            
def load_json(path: str):
    file = None
    with open(path, mode="r") as f:
        file = json.load(f)
        f.close()
    return file


def save_json(obj, path: str):
    with open(path, mode="w") as f:
        json.dump(obj, f, indent=2)
        f.close()