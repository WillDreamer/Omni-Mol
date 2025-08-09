import torch
from .molecule_stm import GNNstm, GNN_graphpred



def build_encoder(config):
    JK = config.JK if "JK" in config.keys() else "last"
    cat_grep = config.cat_grep if "cat_grep" in config.keys() else True
    molecule_node_model = GNNstm(
        num_layer=config.gin_num_layers,
        emb_dim=config.hidden_size,
        gnn_type='gin',
        drop_ratio=config.gin_drop_ratio,
        JK=JK,
    )
    graph_encoder = GNN_graphpred(
        emb_dim=config.hidden_size,
        molecule_node_model=molecule_node_model,
        init_checkpoint="/workspace/molecule_model.pth",
        cat_grep=cat_grep
    )
    ln_graph = LayerNorm(graph_encoder.num_features)
    
    return graph_encoder, ln_graph

class LayerNorm(torch.nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, mask=None):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
