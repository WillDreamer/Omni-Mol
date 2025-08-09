from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
# from timm.layers import LayerNorm, LayerNorm2d
from torch.nn import LayerNorm

from .configuration_llamo import GraphProjectorConfig
from .graph_encoders import LayerNorm

from llamo.chem import mol_to_graphs
from rdkit import Chem

def build_pos_embeds(
    config: GraphProjectorConfig, num_input_tokens: int, vision_hidden_size: int
):
    # pos emb
    if config.pos_emb:
        pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
        nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
    else:
        pos_emb = None

    return pos_emb


def build_eos_tokens(config: GraphProjectorConfig, output_hidden_size: int):
    # think tokens
    num_eos_tokens = config.num_eos_tokens
    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(torch.randn(1, num_eos_tokens, output_hidden_size))
        nn.init.trunc_normal_(eos_tokens, mean=0.0, std=config.initializer_range)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(config: GraphProjectorConfig):
    if config.prenorm:
        prenorm = LayerNorm(config.encoder_hidden_size)
    else:
        prenorm = None
    return prenorm


def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)

class Pooling(nn.Module):
    def __init__(
        self, hidden_size: int, num_queries: int = 32, mode: str = "attn"
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.mode = mode
        
        self.linear = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, x, mask=None):
        y = self.linear(x)/(self.linear.weight.norm()+1e-6)
        if mask is not None:
            added_value = (~mask) * (-1000000)
            y += added_value.unsqueeze(-1)
        topk_values, topk_indices = y.topk(self.num_queries, dim=1)
        x = torch.gather(x,1, topk_indices.repeat(1,1,x.size(-1))) * topk_values.tanh()
        return x


class MultilevelProjector(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        config: GraphProjectorConfig,
        num_query_tokens: int,
        output_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.output_hidden_size = output_hidden_size
        self.num_query_tokens = num_query_tokens
        
        self.cat_num = 0

        self.motif_weight = nn.Linear(config.num_motifs, 300, bias=False)
        self.num_query_tokens = num_query_tokens-4
            
            
        self.motif_cross_attention = nn.MultiheadAttention(300, 4, batch_first=True)
        self.motif_query_tokens = nn.Parameter(torch.randn((4, 300)))

        self.cross_attention = nn.ModuleList([nn.MultiheadAttention(300, 4, batch_first=True)]*6)
        self.query_tokens = nn.Parameter(torch.randn((6, (self.num_query_tokens-self.cat_num)//6, 300)))
        
        # think tokens
        # self.eos_tokens = build_eos_tokens(config, output_hidden_size)
        self.eos_tokens = None
        self.prenorm = build_prenorm(config)
        # self.layer_norm = nn.ModuleList(LayerNorm(output_hidden_size) for _ in range(6))
        self.layer_norm = LayerNorm(config.encoder_hidden_size)

        
        self.build_net()

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor, smiles=None, h_list=None, h_pool_list=None, g_rep=None) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)
                
        num_nodes = x.size(1)
        new_query_tokens = []
        for i in range(6):
            new_query_tokens.append(self.cross_attention[i](self.query_tokens[i].expand(x.size(0), -1, -1).to(x.device), h_list[i][0], h_list[i][0], key_padding_mask=torch.logical_not(h_list[i][1]))[0])
        x = torch.cat(new_query_tokens, 1)

        motifs = []
        for s in smiles:
            try:
                motif = mol_to_graphs(Chem.MolFromSmiles(s)).to(x.device).type(x.dtype)
            except:
                motif = None
            motifs.append(motif)

        for i, motif in enumerate(motifs):
            if motif is None:
                temp_x = self.motif_query_tokens
            else:
                motif_token = self.motif_weight(motif)
                temp_x = self.motif_cross_attention(self.motif_query_tokens, motif_token, motif_token)[0]

            if i == 0:
                add_x = temp_x.unsqueeze(0)
            else:
                add_x = torch.cat((add_x, temp_x.unsqueeze(0)))
        x = torch.cat((add_x, x), dim=1)

        x = self.layer_norm(x)
        query_tokens = self._forward(x)  # (B, L, output_hidden_size)


        B = query_tokens.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([query_tokens, self.eos_tokens.expand(B, -1, -1)], dim=1)
        else:
            x = query_tokens
        return x
    
class MLPMultilevelProjector(MultilevelProjector):
    def build_net(self):
        encoder_hidden_size = self.config.encoder_hidden_size
        # hidden_size = self.config.hidden_size
        output_hidden_size = self.output_hidden_size
        depth = self.config.depth

        self.net = build_mlp(depth, encoder_hidden_size, output_hidden_size)
    
    def _forward(self, x):
        x = self.net(x)
        return x
