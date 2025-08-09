import os
from typing import Any, Dict
import torch
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
import torch.distributed as dist
from peft import LoraConfig, TaskType
from transformers import Adafactor

from llamo.modeling_llamo import LLaMoForConditionalGeneration
from llamo.tokenization_llamo import build_llamo_tokenizer
import time
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np

def caption_evaluate(predictions, targets, tokenizer, text_trunc_length):
    meteor_scores = []
    references = []
    hypotheses = []
    for gt, out in tqdm(zip(targets, predictions)):
        gt_tokens = tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))
        gt_tokens = list(filter(('<pad>').__ne__, gt_tokens))
        gt_tokens = list(filter(('<s>').__ne__, gt_tokens))
        gt_tokens = list(filter(('</s>').__ne__, gt_tokens))
        gt_tokens = list(filter(('[INST]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[/INST]').__ne__, gt_tokens))



        out_tokens = tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))
        out_tokens = list(filter(('<pad>').__ne__, out_tokens))
        out_tokens = list(filter(('<s>').__ne__, out_tokens))
        out_tokens = list(filter(('</s>').__ne__, out_tokens))
        out_tokens = list(filter(('[INST]').__ne__, out_tokens))
        out_tokens = list(filter(('[/INST]').__ne__, out_tokens))
        

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))
    bleu2 *= 100
    bleu4 *= 100

    print('BLEU-2 score:', bleu2)
    print('BLEU-4 score:', bleu4)
    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100
    print('Average Meteor score:', _meteor_score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for gt, out in tqdm(zip(targets, predictions)):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    print('ROUGE score:')
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]) * 100
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]) * 100
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]) * 100
    print('rouge1:', rouge_1)
    print('rouge2:', rouge_2)
    print('rougeL:', rouge_l)
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        

def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    
    model.load_state_dict(state_dict, strict=True)

def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict

class LLaMoStage(pl.LightningModule):
    
    def __init__(self, args, config):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        if not hasattr(args, 'do_sample'):
            args.do_sample = False
        self.caption_eval_epoch = args.caption_eval_epoch
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_len = args.max_len
        self.min_len = args.min_len
        self.llm_tune = args.llm_tune
        self.length_penalty= args.length_penalty
        
        self.new_idx = 0

        self.mllm = LLaMoForConditionalGeneration(config)
        tokenizer_ckpt = config.lm_config.pretrained_tokenizer_name_or_path
        self.tokenizer = build_llamo_tokenizer(tokenizer_ckpt, config.num_query_tokens)


        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})

        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = '<s>'
        
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = '</s>'
        
        self.mllm.resize_token_embeddings(len(self.tokenizer))
        self.mllm.language_model.resize_token_embeddings(len(self.tokenizer))

        self.save_hyperparameters(args)

    def configure_optimizers(self):
        if self.args.optimizer == 'adafactor':
            print('Using adafactor optimizer')
            optimizer = Adafactor(
                self.parameters(),
                lr=1e-3,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False
            )
            self.scheduler = None
        else:
            self.trainer.fit_loop.setup_data()
            warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
            optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
            if self.args.scheduler == 'linear_warmup_cosine_lr':
                self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
            elif self.args.scheduler == 'linear_warmup_step_lr':
                self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
            elif self.args.scheduler == 'None':
                self.scheduler = None
            else:
                raise NotImplementedError()
        return optimizer

    def on_test_epoch_start(self) -> None:
        self.list_predictions = []
        self.list_targets = []
        
    def on_test_epoch_end(self):
        list_predictions = self.list_predictions
        list_targets = self.list_targets

        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]

        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        
        try:
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)
        except RuntimeError:
            all_predictions = [predictions]
            all_targets = [targets]
            
        if self.global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]
            self.save_predictions(all_predictions, all_targets)
            ## fixme: I am not sure if the max length is the same as previous experiments
            bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                caption_evaluate(all_predictions, all_targets, self.tokenizer, self.max_len * 2) 
            self.log("bleu2", bleu2, sync_dist=False)
            self.log("bleu4", bleu4, sync_dist=False)
            self.log("rouge_1", rouge_1, sync_dist=False)
            self.log("rouge_2", rouge_2, sync_dist=False)
            self.log("rouge_l", rouge_l, sync_dist=False)
            self.log("meteor_score", meteor_score, sync_dist=False)

    def save_predictions(self, predictions, targets):
        assert len(predictions) == len(targets)
        with open(os.path.join(self.logger.log_dir, 'predictions.txt'), 'w', encoding='utf8') as f:
            for p, t in zip(predictions, targets):
                line = {'prediction': p, 'target': t}
                f.write(json.dumps(line, ensure_ascii=True) + '\n')

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.mllm.eval()
        graph_values = batch['graph_values']
        input_ids = batch['input_ids']
        smiles = batch['smiles']
        seq_length = batch['seq_length']
        attention_mask = batch['attention_mask']
        label = batch['labels']

        ###============== Captioning Results ===================###
        prediction_ids = self.mllm.generate(
            graph_values=graph_values,
            input_ids=input_ids,
            smiles=smiles,
            seq_length=seq_length,
            attention_mask=attention_mask, 
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            max_new_tokens=self.max_len,
            min_length=self.min_len,
            length_penalty=self.length_penalty
        )
        
        predictions = self.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
        predictions = [pred.strip() for pred in predictions]

        texts = self.tokenizer.batch_decode(label, skip_special_tokens=True)
        texts = [txt.strip() for txt in texts]

        self.list_predictions.append(predictions)
        self.list_targets.append(texts)

        return (predictions, texts)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        self.mllm.eval()
        if dataloader_idx == 0:
            # batch_size = batch['input_ids'].shape[0]
            # outputs = self.mllm(**batch)
            # loss = outputs.loss
            # ###============== Overall Loss ===================###
            # self.log("val molecule loss", float(loss), batch_size=batch_size, sync_dist=True)
            # self.new_idx +=1
            return 0.0
        elif dataloader_idx == 1:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            graph_values = batch['graph_values']
            input_ids = batch['input_ids']
            smiles = batch['smiles']
            attention_mask = batch['attention_mask']
            label = batch['labels']

            ###============== Captioning Results ===================###
            prediction_ids = self.mllm.generate(
                graph_values=graph_values,
                input_ids=input_ids,
                smiles=smiles,
                attention_mask=attention_mask, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_new_tokens=self.max_len,
                min_length=self.min_len,
                length_penalty=self.length_penalty
            )

            predictions = self.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
            predictions = [pred.strip() for pred in predictions]

            texts = self.tokenizer.batch_decode(label, skip_special_tokens=True)
            texts = [txt.strip() for txt in texts]

            
            self.list_predictions.append(predictions)
            self.list_targets.append(texts)

        else:
            raise NotImplementedError
    
    def on_validation_epoch_start(self) -> None:
        self.list_predictions = []
        self.list_targets = []
    
    def on_validation_epoch_end(self) -> None:
    # def validation_epoch_end(self, outputs):
        if (self.current_epoch+1) % self.caption_eval_epoch != 0:
            return 
        # caption_outputs = outputs[1]
        # list_predictions, list_targets = zip(*caption_outputs)
        list_predictions = self.list_predictions
        list_targets = self.list_targets
        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]
        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]
        try:
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)
        except RuntimeError:
            all_predictions = [predictions]
            all_targets = [targets]

        if self.global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]
            self.save_predictions(all_predictions, all_targets)
            ## fixme: I am not sure if the max length is the same as previous experiments
            bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                caption_evaluate(all_predictions, all_targets, self.tokenizer, self.max_len * 2) 
            self.log("bleu2", bleu2, sync_dist=False)
            self.log("bleu4", bleu4, sync_dist=False)
            self.log("rouge_1", rouge_1, sync_dist=False)
            self.log("rouge_2", rouge_2, sync_dist=False)
            self.log("rouge_l", rouge_l, sync_dist=False)
            self.log("meteor_score", meteor_score, sync_dist=False)
        

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        batch_size = batch['input_ids'].shape[0]
        
        ###============== Overall Loss ===================###
        outputs = self.mllm(**batch)
        loss = outputs.loss

        self.log("molecule loss", float(loss), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        # parser.add_argument('--prompt', type=str, default='a molecule of ')
        parser.add_argument('--num_beams', type=int, default=1)
        parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--max_len', type=int, default=512)
        parser.add_argument('--min_len', type=int, default=8)
        parser.add_argument('--length_penalty', type=float, default=1.0)
        
        parser.add_argument('--llm_tune', type=str, default='freeze')
        parser.add_argument('--peft_config', type=str, default=None)
        parser.add_argument('--peft_dir', type=str, default='')

        parser.add_argument('--save_every_n_epochs', type=int, default=10)
        ## quantization
        parser.add_argument('--load_in_8bit', action='store_true', default=False)

        ## lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=int, default=0.1)

        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=5e-5, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=5e-6, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=5e-7, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or linear_warmup_step_lr
        parser.add_argument('--optimizer', type=str, default='adamw', help='type of scheduler')
        parser.add_argument('--stage_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=1)
        
        return parent_parser


