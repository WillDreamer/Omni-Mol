import os

import torch
from peft import PeftConfig, PeftModel
from PIL import Image
from transformers import AutoTokenizer
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from pathlib import Path

from llamo.modeling_llamo import LLaMoForConditionalGeneration
from llamo.processing_llamo import HoneybeeProcessor


def get_processor(config, tokenizer):
    """Model Provider with tokenizer and processor.

    Args:
        pretrained_ckpt (string): The path to pre-trained checkpoint.
        use_bf16 (bool, optional): Whether to use bfloat16 to load the model. (Default: True)
        load_in_8bit(bool, optional): Flag to load model in 8it. (Default: False)

    Returns:
        model: Honeybee Model
        tokenizer: Honeybee (Llama) text tokenizer
        processor: Honeybee processor (including text and image)
    """
    # Load model where base_ckpt is different when the target model is trained by PEFT
    graph_processor = None
    # num_query_tokens = model.config.num_query_tokens
    num_query_tokens = config.num_query_tokens

    num_eos_tokens = getattr(config.graph_projector_config, "num_eos_tokens", 1)
    num_graph_tokens = num_query_tokens + num_eos_tokens
    text_max_len = config.text_max_len
    if "galactica" in config.lm_config.pretrained_lm_name_or_path:
        is_gal = True
    else:
        is_gal = False
        
    graph_only = config.graph_only
    reverse_order = config.reverse_order
    graph_token = config.graph_config.graph_token

    processor = HoneybeeProcessor(
        tokenizer, text_max_len, is_gal, graph_only, reverse_order, num_graph_tokens=num_graph_tokens, graph_token=graph_token
    )

    return processor


def do_generate(
    prompts, image_list, model, tokenizer, processor, use_bf16=False, **generate_kwargs
):
    """The interface for generation

    Args:
        prompts (List[str]): The prompt text
        image_list (List[str]): Paths of images
        model (HoneygraphForConditionalGeneration): HoneygraphForConditionalGeneration
        tokenizer (AutoTokenizer): AutoTokenizer
        processor (HoneybeeProcessor): HoneybeeProcessor
        use_bf16 (bool, optional): Whether to use bfloat16. Defaults to False.

    Returns:
        sentence (str): Generated sentence.
    """
    if image_list:
        images = [Image.open(_) for _ in image_list]
    else:
        images = None
    inputs = processor(text=prompts, images=images)
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)
    sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
    return sentence
