from transformers import PretrainedConfig
from typing import Any, Optional, List
from dataclasses import dataclass, field
from transformers.models.llava import configuration_llava
from model.custom_phi3 import CustomPhi3Config
from transformers.models.llama import LlamaConfig

MODEL_CLS_MAP = {
    "phi3-mini": CustomPhi3Config,
    "Phi3-medium": CustomPhi3Config,
    "llama3-1b": LlamaConfig,
    "llama3-3b": LlamaConfig,
    "llama3-8b": LlamaConfig
}

class MoEConfig(PretrainedConfig):
    model_type="moe_module"
    
    def __init__(
        self,
        train_modules: Optional[List[str]]=None,
        moe_mode: str=None,
        moe_layers_idx: Optional[List[int]]=None,
        ep_size: int=None,
        num_experts: int=None,
        top_k_experts: int=None,
        capacity_factor: float=None,
        eval_capacity_factor: float=None,
        min_capacity: int=None,
        use_residual: bool=None,
        router_aux_loss_coef: float=None,
        enable_lottery_trick: bool=None,
        pruning_percent: float=None,
        **kwargs
    ):
        self.train_modules = train_modules
        self.moe_mode = moe_mode
        self.moe_layers_idx = moe_layers_idx
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.use_residual = use_residual
        self.router_aux_loss_coef = router_aux_loss_coef
        self.enable_lottery_trick = enable_lottery_trick
        self.pruning_percent = pruning_percent
        
        super().__init__(**kwargs)

class GraphConfig(PretrainedConfig):
    model_type = "graph"
    
    def __init__(
        self,
        model_name=None,
        encoder_num_layer=None,
        hidden_size=None,
        encoder_JK=None,
        encoder_drop_ratio=None,
        encoder_gnn_type=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.encoder_num_layer = encoder_num_layer
        self.hidden_size = hidden_size
        self.encoder_JK = encoder_JK
        self.encoder_drop_ratio = encoder_drop_ratio
        self.encoder_gnn_type = encoder_gnn_type
        
class ProjectorConfig(PretrainedConfig):
    model_type = "projector"
    def __init__(
        self,
        projector_type="linear",
        moe_type="type1",
        head_type="gap",
        use_mlp=False,
        level=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.moe_type = moe_type
        self.head_type = head_type
        self.use_mlp = use_mlp
        self.level = level
        

class GraphLlavaConfig(PretrainedConfig):
    model_type = "llava"
    is_composition = False

    def __init__(
        self,
        graph_config=None,
        text_config=None,
        moe_config=None,
        projector_config=None,
        language_backbone_name=None,
        ignore_index=-100,
        image_token_index=-200,
        projector_hidden_act="gelu",
        moe_enable=False,
        projector_type="mulilevel_type1",
        projector_aux_loss_coeff=0.01,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.moe_enable=moe_enable
        self.language_backbone_name=language_backbone_name
        self.projector_type = projector_type

        if graph_config is None:
            graph_config = GraphConfig(
                model_type="himol",
                encoder_num_layer=5,
                hidden_size=300,
                encoder_JK="last",
                encoder_drop_ratio=0.1,
                encoder_gnn_type="gin"
            )
        if isinstance(graph_config, dict):
            graph_config = GraphConfig(**graph_config)
            
        if isinstance(text_config, dict):
            text_config = MODEL_CLS_MAP[language_backbone_name](**text_config)
            
        if isinstance(moe_config, dict):
            moe_config = MoEConfig(**moe_config)
            
        if isinstance(projector_config, dict):
            projector_config = ProjectorConfig(**projector_config)
            
        self.projector_aux_loss_coeff = projector_aux_loss_coeff
        self.graph_config = graph_config
        self.projector_config = projector_config
        self.text_config = text_config
        self.moe_config = moe_config

        super().__init__(**kwargs)