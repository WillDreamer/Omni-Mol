import torch
from torch import nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Optional, Union, Tuple, List
from model.modeling_llava import GraphLlavaForConditionalGeneration, MoECausalLMOutputWithPast

class GraphT5ForConditionalGeneration(GraphLlavaForConditionalGeneration):
    
    def __init__(self, config):
        super().__init__(config)
        # 替换language_model为T5
        self.language_model = T5ForConditionalGeneration(config.text_config)
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        graphs: Optional[torch.FloatTensor] = None,  # 永远是None列表，可以忽略
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        this_task_ids: Optional[int] = None,
        use_task_loss: Optional[bool] = None,
        **kwargs
    ):
        
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)
        
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # 返回与modeling_llava相同格式的输出
        return MoECausalLMOutputWithPast(
            loss=loss,
            moe_loss=None,
            model_loss=loss.detach() if loss is not None else None,
            logits=logits,
            router_aux_coeff=None,
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            hidden_states=outputs.decoder_hidden_states if hasattr(outputs, 'decoder_hidden_states') else None,
            attentions=outputs.decoder_attentions if hasattr(outputs, 'decoder_attentions') else None,
            moe_loss_list=None,
            attention_mask=attention_mask
        )
    
    def _shift_right(self, input_ids):
        """Shift input ids one token to the right"""
        decoder_start_token_id = self.config.text_config.decoder_start_token_id
        pad_token_id = self.config.text_config.pad_token_id
        
        # 获取默认值
        if decoder_start_token_id is None:
            decoder_start_token_id = self.language_model.config.decoder_start_token_id
        if pad_token_id is None:
            pad_token_id = self.language_model.config.pad_token_id
            
        if decoder_start_token_id is None:
            raise ValueError(
                "Make sure to set the decoder_start_token_id attribute of the model's "
                "configuration."
            )
        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # [100, 101, 1]  [decoder_start_token_id, 100, 101]
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id
        
        # 替换-100为pad_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        
        return shifted_input_ids
    
    def generate(self, **kwargs):
        if "graphs" in kwargs:
            graphs = kwargs.pop("graphs")
        
        # T5 specific config
        if self.config.text_config.decoder_start_token_id is None:
            self.config.text_config.decoder_start_token_id = self.config.text_config.pad_token_id
        
        # set decoder_start_token_id
        if "decoder_start_token_id" not in kwargs:
            kwargs["decoder_start_token_id"] = self.config.text_config.decoder_start_token_id
        
        if "pad_token_id" not in kwargs:
            kwargs["pad_token_id"] = self.config.text_config.pad_token_id
        
        if "eos_token_id" not in kwargs:
            kwargs["eos_token_id"] = self.config.text_config.eos_token_id
        
        # ensure attention_mask exists
        if "attention_mask" not in kwargs and "input_ids" in kwargs:
            kwargs["attention_mask"] = kwargs["input_ids"].ne(self.config.text_config.pad_token_id).long()
        
        # T5 does not need to manually set decoder_input_ids, generate will automatically handle it
        # directly call language_model's generate
        return self.language_model.generate(**kwargs)