import torch
from torch_geometric.loader.dataloader import Collater
from pipeline.data_utils.constants import IGNORE_INDEX

def eval_func(text_batch):
    new_batch = []
    for t in text_batch:
        label_mask = t["labels"] != IGNORE_INDEX
        input_mask = t["labels"] == IGNORE_INDEX
        input_ids = t["input_ids"][input_mask]
        attention_mask = t["attention_mask"][input_mask]
        labels = t["labels"][label_mask]
        seq_length = input_ids.size(0)
        new_batch.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "seq_length": seq_length,
        })
    return new_batch

def batchify(batch, tokenizer, max_length: int, use_trunc=True, is_eval=False):
    """collate_fn
    Args:
        batch
        tokenizer
        max_length (int)
        use_trunc (bool)

    NOTE data["image"] can be None (e.g., text-instruction dataset)
    NOTE every batch for each device SHOULD have one image at least;
        if all batch data are text-only ones, error would occurs.
    """
    output_batch = {}
    graph_list = [data["graph"] for data in batch]
    smiles_list = [data["smiles"] for data in batch]

    num_graphs_per_sample = torch.LongTensor([graph is not None for graph in graph_list])
    # 1. remove None images from graph_list
    graph_list = [graph for graph in graph_list if graph is not None]

    # 2. collate for images: [num_images, c, h, w]
    output_batch["graph_values"] = Collater([], [])(graph_list)
    # 3. collate for text
    text_batch = [data["text"] for data in batch]
    if is_eval:
        text_batch = eval_func(text_batch)
    padding = "longest" if use_trunc else "max_length"
    text_batch = tokenizer.batch_collate_pad(
        text_batch,
        padding=padding,
        padding_side="right",
        max_length=max_length,
        is_eval=is_eval,
    )

    # NOTE [bw-compat] Do not use attention mask for training, it will be generated automatically.
    # text_batch.pop("attention_mask")

    output_batch.update({
        **text_batch,
        "num_graphs": num_graphs_per_sample,
    })
    
    # output_batch = {k: v.to("cuda") for k, v in output_batch.items()}
    output_batch.update({"smiles": smiles_list})

    return output_batch