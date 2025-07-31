from args import TrainingArguments, ModelArguments, DataArguments
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoConfig 
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.configs import GraphConfig, GraphLlavaConfig, MoEConfig, ProjectorConfig, PureTextConfig
from model.modeling_llava import GraphLlavaForConditionalGeneration
from model.modeling_moe import DeepseekV2MoE
from model.modeling_textmodel import TextModelForConditionalGeneration
from helpers import no_init_weights
from data_pipe import conversation_lib
import torch
from torch import nn
import os
from deepspeed.moe.layer import MoE

from loggers import WrappedLogger

logger = WrappedLogger(__name__)



def find_all_linear_names(model: nn.Module) -> list:
    """
    Find all modules that is nn.Linear
    Args:
        model (nn.Module): The model used to find linear modules

    Returns:
        list: list of linear modules
    """
    cls = torch.nn.Linear  # we are going to find all nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():  # iterate all modules
        if isinstance(module, cls):  # If it's nn.Linear
            names = name.split('.')  # split the name, name rule: xx.xx
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')  # exclude lm_head
    return list(lora_module_names)


def find_linear_without_moe(model: nn.Module) -> list:
    """Find all linear modules except for graph_tower, mm_projector and lm_head

    Args:
        model (nn.Module): Model

    Returns:
        list: list of found modules
    """
    cls = torch.nn.Linear
    lora_module_names = list()
    exception_list = ("graph_tower", "mm_projector", "lm_head", "deepspeed_moe", "mlp.mlp", "mlp.coefficient", "task_probe", "experts", "moe_gate")
    for name, module in model.named_modules():
        if isinstance(module, cls) and all([exception not in name for exception in exception_list]):
            lora_module_names.append(name)
            
    return lora_module_names

def find_linear_without_moe_gates(model: nn.Module) -> list:
    """Find all linear modules except for graph_tower, mm_projector and lm_head

    Args:
        model (nn.Module): Model

    Returns:
        list: list of found modules
    """
    cls = torch.nn.Linear
    lora_module_names = list()
    exception_list = ("graph_tower", "mm_projector", "lm_head", "deepspeed_moe", "mlp.mlp", "mlp.coefficient", "task_probe", "moe_gate")
    for name, module in model.named_modules():
        if isinstance(module, cls) and all([exception not in name for exception in exception_list]):
            lora_module_names.append(name)
            
    return lora_module_names


def create_stage1_model(model_args: ModelArguments, training_args: TrainingArguments) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Stage 1 model:
    
    - 🔥 mm_projector
    - 🥶 graph tower
    - 🥶 LLM

    Args:
        model_args (ModelArguments): Model arguments
        training_args (TrainingArguments): Training arguments

    Returns:
        tuple[PreTrainedTokenizer, PreTrainedModel]: tokenizer for the specific model and the model itself
    """
    # 1. Init all configs
    graph_config = GraphConfig(
        model_name=model_args.graph_tower,
        encoder_num_layer=model_args.gin_num_layers,
        hidden_size=model_args.gin_hidden_dim,
        encoder_JK='last',
        encoder_drop_ratio=model_args.drop_ratio,
        encoder_gnn_type='gin'
    )
    text_config = AutoConfig.from_pretrained(
        model_args.base_model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        )
    projector_config = ProjectorConfig(
        projector_type=model_args.mm_projector_type,
        moe_type=model_args.projector_moe_type,
        head_type=model_args.head_type,
        use_mlp=model_args.use_mlp,
        level=model_args.level,
        use_head_weight=model_args.use_head_weight,
        num_query=model_args.num_query,
        num_heads=model_args.num_heads
    )
    config = GraphLlavaConfig(
        graph_config, 
        text_config,
        moe_config=None,
        projector_config=projector_config,
        moe_enable=False,
        language_backbone_name=model_args.language_backbone_name,
        projector_aux_loss_coeff=model_args.projector_aux_loss_coeff
        )
    config.use_cache = False
    text_config.use_cache = False
    # 2. Instantiate tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)

    text_config.moe_enable = config.moe_enable = False
    
    with no_init_weights():
        model = GraphLlavaForConditionalGeneration(config)
        
    # 3. Load pre-trained LLM, random init projector and load graph ckpt
    model.load_language_model()
    model.rand_init_projector()
    model.load_graph(model_args.graph_init_checkpoint)
    
    # 4. Set parameters that will be trained
    model.requires_grad_(False)
    model.mm_projector.requires_grad_(True)
    
    return tokenizer, model

def create_partial_model(model_args: ModelArguments, training_args: TrainingArguments) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Stage 2 model
    
    - 🔥 mm_projector
    - 🥶 graph tower
    - 🥶 LLM Backbone 
    - 🔥 LLM Head

    Args:
        model_args (ModelArguments): Model arguments
        training_args (TrainingArguments): Training arguments

    Returns:
        tuple[PreTrainedTokenizer, PreTrainedModel]: tokenizer for the specific model and the model itself
    """
    # 1. Init all configs
    graph_config = GraphConfig(
        model_name=model_args.graph_tower,
        encoder_num_layer=model_args.gin_num_layers,
        hidden_size=model_args.gin_hidden_dim,
        encoder_JK='last',
        encoder_drop_ratio=model_args.drop_ratio,
        encoder_gnn_type='gin'
    )
    # default override to torch.bfloat16 for flash attention
    text_config = AutoConfig.from_pretrained(
        model_args.base_model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        )
    projector_config = ProjectorConfig(
        projector_type=model_args.mm_projector_type,
        use_mlp=model_args.use_mlp,
    )
    config = GraphLlavaConfig(
        graph_config, 
        text_config, 
        projector_config=projector_config,
        moe_enable=model_args.moe_enable,
        language_backbone_name=model_args.language_backbone_name,
        enable_apple_loss=model_args.enable_apple_loss
        )
    config.use_cache = False
    text_config.use_cache = False
    # 2. Instantiate tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)
    
    text_config.moe_enable = config.moe_enable = False
    with no_init_weights():
        model = GraphLlavaForConditionalGeneration(config)
        
    # 3. Load pre-trained LLM, projector and GNN
    model.load_language_model()
    model.load_projector(model_args.pretrain_mm_mlp_adapter)
    model.load_graph(model_args.graph_init_checkpoint)
    
    model.to("cpu")

    model.requires_grad_(False)
    # 5. set parameters, since LoRA freeze all parameters, we activate projector here
    model.mm_projector.requires_grad_(True)

    # head activate
    model.language_model.lm_head.requires_grad_(True)
    return tokenizer, model


def create_lora_model(model_args: ModelArguments, training_args: TrainingArguments) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Stage 2 model
    
    - 🔥 mm_projector
    - 🔥 LoRA
    - 🥶 graph tower
    - 🥶 LLM

    Args:
        model_args (ModelArguments): Model arguments
        training_args (TrainingArguments): Training arguments

    Returns:
        tuple[PreTrainedTokenizer, PreTrainedModel]: tokenizer for the specific model and the model itself
    """
    # 1. Init all configs
    graph_config = GraphConfig(
        model_name=model_args.graph_tower,
        encoder_num_layer=model_args.gin_num_layers,
        hidden_size=model_args.gin_hidden_dim,
        encoder_JK='last',
        encoder_drop_ratio=model_args.drop_ratio,
        encoder_gnn_type='gin'
    )
    # default override to torch.bfloat16 for flash attention
    text_config = AutoConfig.from_pretrained(
        model_args.base_model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        )
    projector_config = ProjectorConfig(
        projector_type=model_args.mm_projector_type,
        use_mlp=model_args.use_mlp,
    )
    config = GraphLlavaConfig(
        graph_config, 
        text_config, 
        projector_config=projector_config,
        moe_enable=model_args.moe_enable,
        language_backbone_name=model_args.language_backbone_name,
        enable_apple_loss=model_args.enable_apple_loss
        )
    config.use_cache = False
    text_config.use_cache = False
    # 2. Instantiate tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)
    
    text_config.moe_enable = config.moe_enable = False
    with no_init_weights():
        model = GraphLlavaForConditionalGeneration(config)
        
    # 3. Load pre-trained LLM, projector and GNN
    model.load_language_model()
    model.load_projector(model_args.pretrain_mm_mlp_adapter)
    model.load_graph(model_args.graph_init_checkpoint)
    
    # 4. Apply LoRA
    # import lora related functions
    from peft import LoraConfig, get_peft_model, PeftModel
    if training_args.resume_from_lora is not None:
        logger.info(f"Resume LoRA adapter from path {training_args.resume_from_lora}", on_rank0=True)
        model = PeftModel.from_pretrained(model, training_args.resume_from_lora, is_trainable=True)
    else:
        lora_config = LoraConfig(  # initailize a LoRA Config
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_linear_without_moe(model),  # add lora to all modules that is nn.Linear
            lora_dropout=training_args.lora_dropout,
            use_rslora=training_args.use_rslora,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if torch.cuda.is_available():
            logger.info("Moving to cuda for faster warping...", on_rank0=True)
            model.to("cuda")
            
        logger.info("Adding LoRA adapters...", on_rank0=True)
        model = get_peft_model(model, lora_config)  # add lora according to lora_config
        
    training_args.lora_enable = True
    model.to("cpu")
    
    # 5. set parameters, since LoRA freeze all parameters, we activate projector here
    model.mm_projector.requires_grad_(True)
        
    return tokenizer, model


def create_moe_lora_model(model_args: ModelArguments, training_args: TrainingArguments) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    # 1. Init all configs
    graph_config = GraphConfig(
        model_name=model_args.graph_tower,
        encoder_num_layer=model_args.gin_num_layers,
        hidden_size=model_args.gin_hidden_dim,
        encoder_JK='last',
        encoder_drop_ratio=model_args.drop_ratio,
        encoder_gnn_type='gin'
    )
    # default override to torch.bfloat16 for flash attention
    text_config = AutoConfig.from_pretrained(
        model_args.base_model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        )
    projector_config = ProjectorConfig(
        projector_type=model_args.mm_projector_type,
        use_mlp=model_args.use_mlp,
    )
    config = GraphLlavaConfig(
        graph_config, 
        text_config, 
        projector_config=projector_config,
        moe_enable=model_args.moe_enable,
        language_backbone_name=model_args.language_backbone_name,
        enable_apple_loss=model_args.enable_apple_loss
        )
    config.use_cache = False
    text_config.use_cache = False
    # 2. Instantiate tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)
    
    with no_init_weights():
        model = GraphLlavaForConditionalGeneration(config)
        
    # 3. Load pre-trained LLM, projector and GNN
    model.load_language_model()
    model.load_projector(model_args.pretrain_mm_mlp_adapter)
    model.load_graph(model_args.graph_init_checkpoint)
    
    # 4. create moe model
    moe_config = MoEConfig(
        moe_mode = model_args.moe_mode,
        moe_layers_idx=model_args.moe_layers_idx,
        ep_size=model_args.ep_size,
        num_experts=model_args.num_experts,
        top_k_experts=model_args.top_k_experts,
        capacity_factor=model_args.capacity_factor,
        eval_capacity_factor=model_args.eval_capacity_factor,
        min_capacity=model_args.min_capacity,
        use_residual=model_args.use_residual,
        router_aux_loss_coef=model_args.router_aux_loss_coef,
        moe_class=model_args.moe_class
    )
    
    model.config.moe_enable = True,
    model.config.text_config.moe_enable = True
    model.config.moe_config = moe_config
    training_args.moe_enable = model_args.moe_enable = True
    
    if torch.cuda.is_available():
        logger.info("Moving to CUDA for faster creation", on_rank0=True)
        model.to("cuda")
        
    model.replace_mlp_with_moe()
    model.to("cpu")
    torch.cuda.empty_cache()
    
    # 5. Apply LoRA
    # import lora related functions
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(  # initailize a LoRA Config
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_linear_without_moe(model),  # do not add lora to any MoE layers, but any layer except that
        use_rslora=training_args.use_rslora,
        use_learnable_alpha=training_args.use_alpha,
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if torch.cuda.is_available():
        logger.info("Moving to cuda for faster warping...", on_rank0=True)
        model.to("cuda")
        
    logger.info("Adding LoRA adapters...", on_rank0=True)
    model = get_peft_model(model, lora_config)  # add lora according to lora_config
    training_args.lora_enable = True
    model.to("cpu")
    
    # 5. set parameters, since LoRA freeze all parameters, we activate projector here
    model.mm_projector.requires_grad_(True)
    # 6. set all moe layers active
    for name, module in model.named_modules():
        if isinstance(module, (MoE, DeepseekV2MoE)):
            module.requires_grad_(True)
    # model.language_model.lm_head.requires_grad_(True)   
    return tokenizer, model
    

def create_lora_moe_model(model_args: ModelArguments, training_args: TrainingArguments) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    # 1. Init all configs
    graph_config = GraphConfig(
        model_name=model_args.graph_tower,
        encoder_num_layer=model_args.gin_num_layers,
        hidden_size=model_args.gin_hidden_dim,
        encoder_JK='last',
        encoder_drop_ratio=model_args.drop_ratio,
        encoder_gnn_type='gin'
    )
    # default override to torch.bfloat16 for flash attention
    text_config = AutoConfig.from_pretrained(
        model_args.base_model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        )
    projector_config = ProjectorConfig(
        projector_type=model_args.mm_projector_type,
        use_mlp=model_args.use_mlp,
    )
    config = GraphLlavaConfig(
        graph_config, 
        text_config, 
        projector_config=projector_config,
        moe_enable=model_args.moe_enable,
        language_backbone_name=model_args.language_backbone_name,
        enable_apple_loss=model_args.enable_apple_loss
        )
    config.use_cache = False
    text_config.use_cache = False
    # 2. Instantiate tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)
    
    with no_init_weights():
        model = GraphLlavaForConditionalGeneration(config)
        
    # 3. Load pre-trained LLM, projector and GNN
    model.load_language_model()
    model.load_projector(model_args.pretrain_mm_mlp_adapter)
    model.load_graph(model_args.graph_init_checkpoint)
    
    # 4. create moe model
    moe_config = MoEConfig(
        moe_mode = model_args.moe_mode,
        moe_layers_idx=model_args.moe_layers_idx,
        ep_size=model_args.ep_size,
        num_experts=model_args.num_experts,
        top_k_experts=model_args.top_k_experts,
        capacity_factor=model_args.capacity_factor,
        eval_capacity_factor=model_args.eval_capacity_factor,
        min_capacity=model_args.min_capacity,
        use_residual=model_args.use_residual,
        router_aux_loss_coef=model_args.router_aux_loss_coef,
        moe_class=model_args.moe_class,
        norm_topk_prob=model_args.norm_topk_prob
    )
    
    model.config.moe_enable = True,
    model.config.text_config.moe_enable = True
    model.config.moe_config = moe_config
    training_args.moe_enable = model_args.moe_enable = True
    
    if torch.cuda.is_available():
        logger.info("Moving to CUDA for faster creation", on_rank0=True)
        model.to("cuda")
        
    model.replace_mlp_with_moe()
    model.to("cpu")
    torch.cuda.empty_cache()
    
    # 5. Apply LoRA
    # import lora related functions
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(  # initailize a LoRA Config
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_linear_without_moe_gates(model),  # do not add lora to any MoE layers, but any layer except that
        use_rslora=training_args.use_rslora,
        use_learnable_alpha=training_args.use_alpha,
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if torch.cuda.is_available():
        logger.info("Moving to cuda for faster warping...", on_rank0=True)
        model.to("cuda")
        
    logger.info("Adding LoRA adapters...", on_rank0=True)
    model = get_peft_model(model, lora_config)  # add lora according to lora_config
    training_args.lora_enable = True
    model.to("cpu")
    
    # 5. set parameters, since LoRA freeze all parameters, we activate projector here
    model.mm_projector.requires_grad_(True)
    
    # 6. activate moe gate
    for name, module in model.named_modules():
        if 'moe_gate' in name:
            module.requires_grad_(True)
    
    # 6. set all moe layers active
    # for name, module in model.named_modules():
    #     if isinstance(module, (MoE, DeepseekV2MoE)):
    #         module.requires_grad_(True)
    # model.language_model.lm_head.requires_grad_(True)   
    return tokenizer, model

def create_puer_text_model(model_args: ModelArguments, training_args: TrainingArguments) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    # 1. Init all configs
    # default override to torch.bfloat16 for flash attention
    text_config = AutoConfig.from_pretrained(
        model_args.base_model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        )
    config = PureTextConfig(
        text_config, 
        moe_enable=model_args.moe_enable,
        language_backbone_name=model_args.language_backbone_name,
        enable_apple_loss=model_args.enable_apple_loss
    )
    config.use_cache = False
    text_config.use_cache = False
    # 2. Instantiate tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)
    
    with no_init_weights():
        model = TextModelForConditionalGeneration(config)
        
    # 3. Load pre-trained LLM
    model.load_language_model()
    
    # 4. create moe model
    moe_config = MoEConfig(
        moe_mode = model_args.moe_mode,
        moe_layers_idx=model_args.moe_layers_idx,
        ep_size=model_args.ep_size,
        num_experts=model_args.num_experts,
        top_k_experts=model_args.top_k_experts,
        capacity_factor=model_args.capacity_factor,
        eval_capacity_factor=model_args.eval_capacity_factor,
        min_capacity=model_args.min_capacity,
        use_residual=model_args.use_residual,
        router_aux_loss_coef=model_args.router_aux_loss_coef,
        moe_class=model_args.moe_class,
        norm_topk_prob=model_args.norm_topk_prob
    )
    
    model.config.moe_enable = True,
    model.config.text_config.moe_enable = True
    model.config.moe_config = moe_config
    training_args.moe_enable = model_args.moe_enable = True
    
    if torch.cuda.is_available():
        logger.info("Moving to CUDA for faster creation", on_rank0=True)
        model.to("cuda")
        
    model.replace_mlp_with_moe()
    model.to("cpu")
    torch.cuda.empty_cache()
    
    # 5. Apply LoRA
    # import lora related functions
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(  # initailize a LoRA Config
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_linear_without_moe_gates(model),  # do not add lora to any MoE layers, but any layer except that
        use_rslora=training_args.use_rslora,
        use_learnable_alpha=training_args.use_alpha,
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if torch.cuda.is_available():
        logger.info("Moving to cuda for faster warping...", on_rank0=True)
        model.to("cuda")
        
    logger.info("Adding LoRA adapters...", on_rank0=True)
    model = get_peft_model(model, lora_config)  # add lora according to lora_config
    training_args.lora_enable = True
    model.to("cpu")
    
    # 6. activate moe gate
    for name, module in model.named_modules():
        if 'moe_gate' in name:
            module.requires_grad_(True)
    
    # 6. set all moe layers active
    # for name, module in model.named_modules():
    #     if isinstance(module, (MoE, DeepseekV2MoE)):
    #         module.requires_grad_(True)
    # model.language_model.lm_head.requires_grad_(True)   
    model.enable_input_require_grads()
    return tokenizer, model
    


def load_lora_model(
    model_path: str,
    language_backbone: str,
    graph_path: str,
    use_flash_attn: bool,
    task_embed_path: str = None,
    **kwargs
) -> tuple[PreTrainedTokenizer, GraphLlavaForConditionalGeneration]:
    """Load a LoRA fine-tuned model from stage 2

    ## Args:
        model_path (str): path to the lora fine-tuned folder(the one contains adapter_model.safetensors)
        language_backbone (str): path to the language backbone(e.g., phi-3 mini, llama-3.2)
        graph_path (str): path to graph checkpoint(e.g. himol_encoder.pth)
        use_flash_attn (bool): Whether to use flash attention

    ## Raises:
        NotImplementedError: If no non_lora_trainables.bin exists in the model_path, something happend to the saving
        or the parameter activation in the training, please check!

    ## Returns:
        tuple[PreTrainedTokenizer, GraphLlavaForConditionalGeneration]: tokenizer for the specific model and the model itself
    """
    # 1. Get config from the model_path folder
    config = GraphLlavaConfig.from_pretrained(model_path)
    if use_flash_attn:
        config._attn_implementation = "flash_attention_2"
        config.text_config._attn_implementation = "flash_attention_2"
    with no_init_weights():
        model = GraphLlavaForConditionalGeneration(config)
        
    tokenizer = AutoTokenizer.from_pretrained(language_backbone)
    # 2. Load language model, graph ckpt
    model.load_language_model()
    model.load_graph(graph_path)
    
    # 3. Load mm_projector
    logger.info('Loading additional LLaVA weights...', on_rank0=True)
    # Read and process non-lora-trainables, specifically, mm projector
    if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(
            os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu', weights_only=True)
        logger.info(f"Non-lora trainables: {non_lora_trainables.keys()}", on_rank0=True)
    else:
        logger.info("No Non-lora weights detected!", on_rank0=True)
        raise NotImplementedError
        
    non_lora_trainables = {k.split("base_model.model.")[1]: v for k, v in non_lora_trainables.items()}
        
    model.load_state_dict(non_lora_trainables, strict=False)
    
    if task_embed_path is not None:
        model.load_task_embed(task_embed_path)
        
    # 4. Load LoRA weights and merge LoRA
    from peft import PeftModel
    logger.info('Loading LoRA weights...', on_rank0=True)
    model = PeftModel.from_pretrained(model, model_path)
    logger.info("Moving to CUDA", on_rank0=True)
    model.to(torch.device("cuda"))
    logger.info('Merging LoRA weights...', on_rank0=True)
    model = model.merge_and_unload()
    logger.info("Moving back to CPU", on_rank0=True)
    model.to(torch.device("cpu"))
    logger.info('Model is loaded...', on_rank0=True)
    torch.cuda.empty_cache()
        
    return tokenizer, model


def load_moe_lora_model(
    model_path: str,
    language_backbone: str,
    graph_path: str,
    use_flash_attn: bool,
    *args,
    **kwargs
):
    # 1. Build base language model
    config = GraphLlavaConfig.from_pretrained(model_path)
    if use_flash_attn:
        config._attn_implementation = "flash_attention_2"
        config.text_config._attn_implementation = "flash_attention_2"
    with no_init_weights():
        model = GraphLlavaForConditionalGeneration(config)
        
    tokenizer = AutoTokenizer.from_pretrained(language_backbone)
    model.load_language_model()
    model.load_graph(graph_path)
    
    # 2. Expand to MoE
    logger.info("Moving to CUDA")
    model.to(torch.device("cuda"))
    model.replace_mlp_with_moe()
    logger.info("Moving back to CPU")
    model.to(torch.device("cpu"))
    
    # 3. Load MoE and projctor weights
    logger.info('Loading additional LLaVA weights...')
    if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(
            os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        logger.info(f"Non-lora trainables: {non_lora_trainables.keys()}", on_rank0=True)
    else:
        logger.info("No Non-lora weights detected!")
        raise NotImplementedError
    non_lora_state_dict = {k.split("base_model.model.")[1]: v for k, v in non_lora_trainables.items()}
    
    model.load_state_dict(non_lora_state_dict, strict=False)
    
    # 4. load and merge lora
    from peft import PeftModel
    logger.info('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    logger.info("Moving to CUDA")
    model.to(torch.device("cuda"))
    logger.info('Merging LoRA weights...')
    model = model.merge_and_unload()
    logger.info("Moving back to CPU")
    model.to(torch.device("cpu"))
    logger.info('Model is loaded...')
    torch.cuda.empty_cache()

    # 5. deepspeed engine
    if any(isinstance(module, MoE) for _, module in model.named_modules()):
        import deepspeed
        deepspeed.init_distributed(dist_backend='nccl')
        # Initialize the DeepSpeed-Inference engine
        ds_engine = deepspeed.init_inference(model,
                                                # mp_size=2,
                                                # dtype=torch.half,
                                                checkpoint=None,
                                                replace_with_kernel_inject=False)
        model = ds_engine.module
        
    return tokenizer, model

def load_moe_lora_model_sequential(
    model_args: ModelArguments, training_args: TrainingArguments
):
    # 1. Build base language model
    config = GraphLlavaConfig.from_pretrained(model_args.model_name_or_path)

    with no_init_weights():
        model = GraphLlavaForConditionalGeneration(config)
        
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)
    model.load_language_model()
    model.load_graph(model_args.graph_init_checkpoint)
    
    # 2. Expand to MoE
    logger.info("Moving to CUDA")
    model.to(torch.device("cuda"))
    model.replace_mlp_with_moe()
    logger.info("Moving back to CPU")
    model.to(torch.device("cpu"))
    
    # 3. Load MoE and projctor weights
    logger.info('Loading additional LLaVA weights...')
    if os.path.exists(os.path.join(model_args.model_name_or_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(
            os.path.join(model_args.model_name_or_path, 'non_lora_trainables.bin'), map_location='cpu')
        logger.info(f"Non-lora trainables: {non_lora_trainables.keys()}", on_rank0=True)
    else:
        logger.info("No Non-lora weights detected!")
        raise NotImplementedError
    non_lora_state_dict = {k.split("base_model.model.")[1]: v for k, v in non_lora_trainables.items()}
    
    model.load_state_dict(non_lora_state_dict, strict=False)
    
    # 4. load and merge lora
    from peft import PeftModel
    logger.info('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_args.model_name_or_path)
    torch.cuda.empty_cache()
    training_args.lora_enable = True
    for name, param in model.named_parameters():
        if any(keyword in name.lower() for keyword in ['mm_projector','lora', 'moe', 'expert', 'lm_head']):
            param.requires_grad = True
        else:
            param.requires_grad = False  # 可选：其余参数显式冻结
    model.enable_input_require_grads()  
    return tokenizer, model

def load_partial_model(
    model_path: str,
    language_backbone: str,
    graph_path: str,
    use_flash_attn: bool,
    task_embed_path: str = None,
    **kwargs
) -> tuple[PreTrainedTokenizer, GraphLlavaForConditionalGeneration]:
    """Load a LoRA fine-tuned model from stage 2

    ## Args:
        model_path (str): path to the lora fine-tuned folder(the one contains adapter_model.safetensors)
        language_backbone (str): path to the language backbone(e.g., phi-3 mini, llama-3.2)
        graph_path (str): path to graph checkpoint(e.g. himol_encoder.pth)
        use_flash_attn (bool): Whether to use flash attention

    ## Raises:
        NotImplementedError: If no non_lora_trainables.bin exists in the model_path, something happend to the saving
        or the parameter activation in the training, please check!

    ## Returns:
        tuple[PreTrainedTokenizer, GraphLlavaForConditionalGeneration]: tokenizer for the specific model and the model itself
    """
    # 1. Get config from the model_path folder
    config = GraphLlavaConfig.from_pretrained(model_path)
    if use_flash_attn:
        config._attn_implementation = "flash_attention_2"
        config.text_config._attn_implementation = "flash_attention_2"
    with no_init_weights():
        model = GraphLlavaForConditionalGeneration(config)
        
    tokenizer = AutoTokenizer.from_pretrained(language_backbone)
    # 2. Load language model, graph ckpt
    model.load_language_model()
    model.load_graph(graph_path)
    
    # 3. Load mm_projector
    logger.info('Loading additional LLaVA weights...', on_rank0=True)
    # Read and process non-lora-trainables, specifically, mm projector
    if os.path.exists(os.path.join(model_path, 'trainables.bin')):
        trainables = torch.load(
            os.path.join(model_path, 'trainables.bin'), map_location='cpu', weights_only=True)
        logger.info(f"trainables: {trainables.keys()}", on_rank0=True)
    else:
        logger.info("No trainable weights detected!", on_rank0=True)
        raise NotImplementedError
        
    model.load_state_dict(trainables, strict=False)
        
    return tokenizer, model

def load_pure_text_model(
    model_path: str,
    language_backbone: str,
    graph_path: str,
    use_flash_attn: bool,
    task_embed_path: str = None,
    **kwargs
) -> tuple[PreTrainedTokenizer, GraphLlavaForConditionalGeneration]:
    # 1. Build base language model
    config = PureTextConfig.from_pretrained(model_path)
    if use_flash_attn:
        config._attn_implementation = "flash_attention_2"
        config.text_config._attn_implementation = "flash_attention_2"
    with no_init_weights():
        model = TextModelForConditionalGeneration(config)
        
    tokenizer = AutoTokenizer.from_pretrained(language_backbone)
    model.load_language_model()
    
    # 2. Expand to MoE
    logger.info("Moving to CUDA")
    model.to(torch.device("cuda"))
    model.replace_mlp_with_moe()
    logger.info("Moving back to CPU")
    model.to(torch.device("cpu"))
    
    # 3. Load MoE and projctor weights
    logger.info('Loading additional LLaVA weights...')
    if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(
            os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        logger.info(f"Non-lora trainables: {non_lora_trainables.keys()}", on_rank0=True)
    else:
        logger.info("No Non-lora weights detected!")
        raise NotImplementedError
    non_lora_state_dict = {k.split("base_model.model.")[1]: v for k, v in non_lora_trainables.items()}
    
    model.load_state_dict(non_lora_state_dict, strict=False)
    
    # 4. load and merge lora
    from peft import PeftModel
    logger.info('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    logger.info("Moving to CUDA")
    model.to(torch.device("cuda"))
    logger.info('Merging LoRA weights...')
    model = model.merge_and_unload()
    logger.info("Moving back to CPU")
    model.to(torch.device("cpu"))
    logger.info('Model is loaded...')
    torch.cuda.empty_cache()

    # 5. deepspeed engine
    if any(isinstance(module, MoE) for _, module in model.named_modules()):
        import deepspeed
        deepspeed.init_distributed(dist_backend='nccl')
        # Initialize the DeepSpeed-Inference engine
        ds_engine = deepspeed.init_inference(model,
                                                # mp_size=2,
                                                # dtype=torch.half,
                                                checkpoint=None,
                                                replace_with_kernel_inject=False)
        model = ds_engine.module
        
    return tokenizer, model
    
    
MODEL_STAGE_MAP = {
    "stage1": create_stage1_model,
    "partial":create_partial_model,
    "lora": create_lora_model,
    "moe+lora": create_moe_lora_model,  #### pure moe
    "loramoe": create_lora_moe_model,  ##### moe with lora
    "sequential": load_moe_lora_model_sequential,
    "puretext": create_puer_text_model  # pure text model
}

def create_model(
    model_args: ModelArguments, 
    data_args: DataArguments, 
    training_args: TrainingArguments
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    # 1. Create tokenizer, model
    tokenizer, model = MODEL_STAGE_MAP[training_args.training_recipe](model_args, training_args)
    
    # 2. Set correct conversation template
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    if tokenizer.pad_token is None and model_args.version == "llama3":
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    elif model_args.version == "tinyllama":
        tokenizer.pad_token = tokenizer.unk_token
    logger.info(f"Using conversation template of {model_args.version}", on_rank0=True)
    logger.info(f"Conversation template: {conversation_lib.default_conversation}", on_rank0=True)
    
    # 3. Align arguments
    data_args.graph_tower = model_args.graph_tower
    training_args.moe_enable = model_args.moe_enable
    training_args.router_aux_coeff = model_args.router_aux_loss_coef
    if "type" in model_args.mm_projector_type:
        training_args.moe_enable = True
        
    return tokenizer, model
