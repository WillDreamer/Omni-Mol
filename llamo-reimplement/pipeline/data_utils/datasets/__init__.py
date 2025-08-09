from .base import BaseDataset

DATASET_CLASS_DICT = {
    "Base": BaseDataset,
}

def load_dataset(
    dset_name, input_path, tokenizer, max_length, **kwargs
):
    dataset_class = DATASET_CLASS_DICT[dset_name]
    dset = dataset_class(input_path, tokenizer, max_length, **kwargs)


    return dset