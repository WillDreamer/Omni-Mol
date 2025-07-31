from model.modeling_llama import CustomLlamaForCausalLM
from transformers.models.llama import LlamaConfig

MODEL_CLS_MAP = {
    "llama": CustomLlamaForCausalLM,
}

MODEL_CONF_MAP = {
    "llama": LlamaConfig
}