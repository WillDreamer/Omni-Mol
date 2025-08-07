import torch
from torch_geometric.data import Batch, Data
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
from typing import Sequence, Dict, Tuple, List

IGNORE_INDEX = -100

@dataclass       
class GraphDatasetCollator(object):
    """Collate graph-QA examples for supervised fine-tuning."""
    
    tokenizer: PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        task_ids = [instance["this_task_ids"] for instance in instances]
        
        is_t5 = "decoder_attention_mask" in instances[0]
        batch = {}
        
        if is_t5:
            attention_mask = [instance["attention_mask"] for instance in instances]
            decoder_attention_mask = [instance["decoder_attention_mask"] for instance in instances]
            
            batch = {
                'input_ids': self._pad_sequence(input_ids, self.tokenizer.pad_token_id),
                'attention_mask': self._pad_sequence(attention_mask, 0),
                'labels': self._pad_sequence(labels, IGNORE_INDEX),
                'decoder_attention_mask': self._pad_sequence(decoder_attention_mask, 0),
                'this_task_ids': torch.cat(task_ids, dim=0)
            }
            
            max_len = self.tokenizer.model_max_length
            batch['input_ids'] = batch['input_ids'][:, :max_len]
            batch['attention_mask'] = batch['attention_mask'][:, :max_len]
            batch['labels'] = batch['labels'][:, :max_len]
            batch['decoder_attention_mask'] = batch['decoder_attention_mask'][:, :max_len]
            
        else:
            input_ids = self._pad_sequence(input_ids, self.tokenizer.pad_token_id)
            labels = self._pad_sequence(labels, IGNORE_INDEX)
            task_ids = torch.cat(task_ids, dim=0)
            
            batch = {
                'input_ids': input_ids[:, :self.tokenizer.model_max_length],
                'labels': labels[:, :self.tokenizer.model_max_length],
                'attention_mask': input_ids[:, :self.tokenizer.model_max_length].ne(self.tokenizer.pad_token_id),
                'this_task_ids': task_ids
            }
        
        if 'graphs' in instances[0]:
            graph_batch = []
            for instance in instances:
                if instance["graphs"] is not None:
                    graph_batch.append(self._convert_dict_to_Data(instance["graphs"]))
                else:
                    graph_batch.append(None)
            batch["graphs"] = graph_batch
            
        return batch
    
    def _extract_tensors(self, instances, keys: Tuple[str, str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return tuple([instance[key] for instance in instances] for key in keys)

    def _pad_sequence(self, sequence: List[torch.Tensor], padding_value: int) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(sequence, batch_first=True, padding_value=padding_value)

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
