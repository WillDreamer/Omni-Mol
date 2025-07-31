from model.modeling_llama import CustomLlamaForCausalLM
from transformers.models.llama import LlamaConfig
from transformers.models.t5 import T5ForConditionalGeneration, T5Config

MODEL_CLS_MAP = {
    "llama": CustomLlamaForCausalLM,
    "t5": T5ForConditionalGeneration,  # TODO: add t5
}

MODEL_CONF_MAP = {
    "llama": LlamaConfig,
    "t5": T5Config  # TODO: add t5
}