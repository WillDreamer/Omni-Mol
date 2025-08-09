# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#### delete after debugging ####
import argparse
import torch
from pytorch_lightning import LightningDataModule
# import torch_geometric
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from .datasets import load_dataset
from functools import partial
from ..collate import batchify
import re


class LLaMoDM(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 4,
        batch_size: int = 256,
        root_train: str = 'data/PubChem324kV2/',
        root_eval: str = 'data/PubChem324kV2/',
        max_length: int = 256,
        tokenizer=None,
        args=None,
        config=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.tokenizer=tokenizer

        self.pretrain_dataset = load_dataset("Base", root_train+f'train.pt', self.tokenizer, self.max_length)
        self.train_dataset = load_dataset("Base", root_train+f'train.pt', self.tokenizer, self.max_length)
        self.val_dataset = load_dataset("Base", root_eval+f'test.pt', self.tokenizer, self.max_length)
        self.test_dataset = load_dataset("Base", root_eval+f'test.pt', self.tokenizer, self.max_length)


    def train_dataloader(self):
        collate_fn = partial(
            batchify,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            use_trunc=True,
            is_eval=False,
        )

        if self.mode == 'train':
            # sampler = DistributedSampler(self.pretrain_dataset) if not isinstance(self.pretrain_dataset, IterableDataset) else None
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=collate_fn,
            )
        else:
            raise NotImplementedError
        return loader

    
    def val_dataloader(self):
        collate_fn = partial(
            batchify,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            use_trunc=True,
            is_eval=True,
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=collate_fn,
        )
        return [val_loader, test_loader]
    
    def test_dataloader(self):
        collate_fn = partial(
            batchify,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            use_trunc=True,
            is_eval=True,
        )
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=collate_fn,
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--root_train', type=str, default='data/PubChemDataset_v4')
        parser.add_argument('--root_eval', type=str, default='data/PubChemDataset_v4')
        parser.add_argument('--filtered_cid_path', type=str, default=None)


        return parent_parser
    