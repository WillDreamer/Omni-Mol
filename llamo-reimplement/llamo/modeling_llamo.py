from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModelForCausalLM
from transformers.utils import (
    ModelOutput,
    logging,
)

from llamo.configuration_llamo import GraphConfig
from llamo.graph_encoders import build_encoder
from torch_geometric.data import Data
from utils import check_local_file

from .projectors import MLPMultilevelProjector

logger = logging.get_logger(__name__)


@dataclass
class LLaMoForConditionalGenerationModelOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        graph_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.

        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    graph_outputs: Optional[torch.FloatTensor] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["graph_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


def get_ltor_masks_and_position_ids_from_embeddings(data):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()[:2]

    # Attention mask (lower triangular).
    att_mask_batch = 1
    attention_mask = torch.tril(
        torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
    ).view(att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size()[:2], dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data[..., 0])

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def apply_delta(base_model, delta_model_name_or_path):
    # Reference: fastchat/model/apply_delta.py from https://github.com/lm-sys/FastChat (vicuna)
    print(f"Loading the delta weights from {delta_model_name_or_path}")
    local_files_only, delta_file_name = check_local_file(delta_model_name_or_path)
    delta, loading_info = AutoModelForCausalLM.from_pretrained(
        delta_file_name,
        local_files_only=local_files_only,
        output_loading_info=True,
    )
    print("[Loading info for delta model] \n", loading_info)
    print("Applying the delta ...")
    for name, param in tqdm(base_model.state_dict().items(), desc="Applying delta"):
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name]

    return base_model


class GraphPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = GraphConfig
    base_model_prefix = "mllm"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"language_model.encoder.embed_tokens.weight",
        r"language_model.decoder.embed_tokens.weight",
        r"language_model.lm_head.weight",
    ]
    _no_split_modules = [
        "LlamaDecoderLayer",
        "LlamaForCausalLM",
        "Parameter",
    ]
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_range
        if (
            isinstance(module, nn.Embedding)
            or isinstance(module, nn.Linear)
        ):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.Parameter):
            raise ValueError
            nn.init.trunc_normal_(module.data, mean=0.0, std=factor)

    # def _set_gradient_checkpointing(self, module, value=False):
        # from transformers.models.clip import modeling_clip

        # if isinstance(module, modeling_clip.CLIPEncoder):
        #     module.gradient_checkpointing = value


def get_media_indices(my_list):
    if isinstance(my_list, torch.Tensor):
        my_list = my_list.cpu().tolist()
    result = []
    for i in range(len(my_list)):
        if i == 0 and my_list[i] < 0:
            result.append(i)
        elif my_list[i] != my_list[i - 1] and my_list[i] < 0:
            result.append(i)
    return result


class LLaMoForConditionalGeneration(GraphPreTrainedModel):
    config_class = GraphConfig
    main_input_name = "graph_values"

    def build_projector(self, config: GraphConfig):
        """Build projector (abstractor) and query_tokens (optionally for resampler)"""
        proj_config = config.graph_projector_config
        proj_type = proj_config.projector_type
        output_hidden_size = config.text_config.hidden_size  # LM hidden size
        num_query_tokens = config.num_query_tokens

        self.abstractor = {
            "MLPMultilevelProjector": MLPMultilevelProjector,
        }[
            proj_type
        ](proj_config, num_query_tokens, output_hidden_size)

    def __init__(self, config: GraphConfig):
        super().__init__(config)
        self.config = config
        
        self.graph_model, self.ln_graph_model = build_encoder(config.graph_config)
        if not config.tune_gnn:
            for param in self.graph_model.parameters():
                param.requires_grad = False
            self.graph_model = self.graph_model.eval()
        # visual projector
        proj_config = config.graph_projector_config
        self.proj_type = proj_config.projector_type
        self.num_query_tokens = config.num_query_tokens
        self.build_projector(config)

        # language model (decoder)
        lm_local_files_only, lm_file_name = check_local_file(
            config.lm_config.pretrained_lm_name_or_path
        )
        language_model = AutoModelForCausalLM.from_pretrained(
            lm_file_name,
            local_files_only=lm_local_files_only,
            torch_dtype=torch.bfloat16
        )

        if config.lm_config.delta_model_name_or_path is not None:
            apply_delta(language_model, config.lm_config.delta_model_name_or_path)
        self.language_model = language_model

        if config.llm_tune == 'lora':
            if "peft_dir" in config.to_dict() and config.peft_dir:
                self.language_model = PeftModel.from_pretrained(self.language_model, config.peft_dir, is_trainable=True)                
            else:
                if "peft_config" in config.to_dict() and config.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(config.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, lora_dropout=0.05, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
                self.peft_config = peft_config
        elif config.llm_tune == 'freeze':
            for name, param in self.language_model.named_parameters():
                if "embed_tokens" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            # for param in self.language_model.embed_tokens.parameters():
                # param.requires_grad = True  # Unfreeze embeddings
        elif config.llm_tune == 'full':
            pass
        else:
            raise ValueError(f"Invalid llm_tune: {config.llm_tune}")

        # Initialize weights and apply final processing
        # Here, weights of abstractor (HoneybeeVisualProjectorModel) is initialized
        self.post_init()
        self.main_input_name = "input_ids"
        from transformers import GenerationConfig

        self.generation_config = GenerationConfig(
            max_length=512,
            do_sample=True,
            pad_token_id=0,
            unk_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )

    def init_peft(self):
        self.language_model = get_peft_model(self.language_model, self.peft_config)
        self.language_model.print_trainable_parameters()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if (
            len(hf_device_map) > 1
            and "language_model" not in hf_device_map
            and torch.cuda.device_count() > 1
        ):
            # warn users about unexpected behavior when using multi-GPU + Honeybee + `accelerate`.
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    def _get_input_dtype(self):
        dtype = self.graph_model.get_dtype()

        return dtype

    def forward_and_project_graph(self, graph_values, smiles=None):
        """Forward graph_values & project (abstract) the visual features to LLM embedding space."""
        assert graph_values is not None

        # =================================================== #
        # Forward graph model
        # =================================================== #
        g_outputs, g_mask, h_list, h_pool_list, graph_representation = self.graph_model(graph_values)

        graph_embeds = self.ln_graph_model(g_outputs, g_mask)  # [B, num_patches+1, dim]
        
        query_features = self.abstractor(graph_embeds, smiles, h_list, h_pool_list, graph_representation)

        # query_features: [B, L, dim]
        return query_features

    def forward(
        self,
        graph_values: Data,
        input_ids: torch.FloatTensor,
        num_graphs,
        smiles=None,
        seq_length: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        labels_attention_mask: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LLaMoForConditionalGenerationModelOutput]:
        r"""
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # get text embedding
        text_tokens_ = input_ids.clone()
        batch_size = input_ids.shape[0]

        media_token_indices = [
            # [:-1] since we would not use the last token for embedding
            get_media_indices(text_tokens_[i])
            for i in range(batch_size)
        ]
        text_tokens_[text_tokens_ < 0] = 0  # Not used
        text_embeds = self.get_input_embeddings()(text_tokens_)  # Temporally Embedding
        if hasattr(self.language_model, "transformer") and hasattr(
            self.language_model.transformer, "word_embeddings_layernorm"
        ):
            text_embeds = self.language_model.transformer.word_embeddings_layernorm(text_embeds)

        if graph_values is not None:
            query_features = self.forward_and_project_graph(graph_values, smiles=smiles)
            #TODO: have to check
            graph_seq_length = query_features.shape[1]  # [B, L, lm_dim]
        num_graphs_per_sample = num_graphs.long().cpu().tolist()

        text_chunk_embeds = []
        input_chunk_attns = []
        graph_idx = 0
        # sanity check (-1 is image token)
        n_graph_tokens = (input_ids == -1).sum(1)
        assert (
            (n_graph_tokens == num_graphs * graph_seq_length).all().item()
        ), f"Expected #img_tokens={n_graph_tokens}, but got {num_graphs * graph_seq_length}"

        for b in range(batch_size):
            start = 0
            embeds = []
            attns = []
            if len(media_token_indices[b]) > 0:
                for i, pos in enumerate(media_token_indices[b]):
                    if pos > start:
                        embeds.append(text_embeds[b, start:pos])  # add tokens before visual tokens
                        attns.append(attention_mask[b, start:pos])
                    embeds.append(query_features[graph_idx + i])  # add visual tokens
                    graph_embed_attn_mask = torch.ones(
                        query_features[graph_idx + i].shape[0], device=text_embeds.device
                    )
                    attns.append(graph_embed_attn_mask)
                    start = pos + graph_seq_length
            if start < text_embeds.shape[1]:
                embeds.append(text_embeds[b, start:])  # add instruction & response
                attns.append(attention_mask[b, start:])

            graph_idx += num_graphs_per_sample[b]
            text_chunk_embeds.append(torch.cat(embeds, dim=0))
            input_chunk_attns.append(torch.cat(attns, dim=0))
        # Actual Input Embeddings
        input_embeds = torch.stack(text_chunk_embeds, dim=0)
        attention_mask = torch.stack(input_chunk_attns, dim=0)

        # Forward into GPT
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            output_attentions=self.config.output_attentions,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        graph_values: Data = None,
        input_ids: Optional[torch.LongTensor] = None,
        smiles = None,
        seq_length: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        isdecoder=True,
        is_null_image=None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            graph_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if input_ids is None:
            return self.language_model.generate(attention_mask=attention_mask, **generate_kwargs)

        if attention_mask is None:
            attention_mask = input_ids.new_ones(*input_ids.shape)
        
        text_tokens_ = input_ids.clone()
        batch_size = input_ids.shape[0]

        media_token_indices = [
            # [:-1] since we would not use the last token for embedding
            get_media_indices(text_tokens_[i])
            for i in range(batch_size)
        ]
        text_tokens_[text_tokens_ < 0] = 0  # Not used
        text_embeds = self.get_input_embeddings()(text_tokens_)  # Temporally Embedding
        if hasattr(self.language_model, "transformer") and hasattr(
            self.language_model.transformer, "word_embeddings_layernorm"
        ):
            text_embeds = self.language_model.transformer.word_embeddings_layernorm(text_embeds)

        if graph_values is not None:
            query_features = self.forward_and_project_graph(graph_values, smiles=smiles)
            #TODO: have to check
            graph_seq_length = query_features.shape[1]  # [B, L, lm_dim]
        num_graphs_per_sample = [len(x) for x in media_token_indices]

        text_chunk_embeds = []
        input_chunk_attns = []
        graph_idx = 0
        # sanity check (-1 is image token)
        n_graph_tokens = (input_ids == -1).sum(1)
        num_graphs = torch.as_tensor(num_graphs_per_sample, device=input_ids.device)
        assert (
            (n_graph_tokens == num_graphs * graph_seq_length).all().item()
        ), f"Expected #img_tokens={n_graph_tokens}, but got {num_graphs * graph_seq_length}"

        for b in range(batch_size):
            start = 0
            embeds = []
            attns = []
            if len(media_token_indices[b]) > 0:
                for i, pos in enumerate(media_token_indices[b]):
                    if pos > start:
                        embeds.append(text_embeds[b, start:pos])  # add tokens before visual tokens
                        attns.append(attention_mask[b, start:pos])
                    embeds.append(query_features[graph_idx + i])  # add visual tokens
                    graph_embed_attn_mask = torch.ones(
                        query_features[graph_idx + i].shape[0], device=text_embeds.device
                    )
                    attns.append(graph_embed_attn_mask)
                    start = pos + graph_seq_length
            if start < text_embeds.shape[1]:
                embeds.append(text_embeds[b, start:])  # add instruction & response
                attns.append(attention_mask[b, start:])

            graph_idx += num_graphs_per_sample[b]
            text_chunk_embeds.append(torch.cat(embeds, dim=0))
            input_chunk_attns.append(torch.cat(attns, dim=0))
        # Actual Input Embeddings
        input_embeds = torch.stack(text_chunk_embeds, dim=0)
        attention_mask = torch.stack(input_chunk_attns, dim=0)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        graph_values=None,
        smiles=None,
        past_key_values=None,
        attention_mask=None,
        **model_kwargs,
    ):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model,
        # the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {
            "input_ids": input_ids,
            "graph_values": graph_values,
            "attention_mask": attention_mask,
            "smiles": smiles,
            "is_decoder": True,
        }

