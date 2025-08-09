import os
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins.environments import ClusterEnvironment

from pipeline.data_utils.llamo_dm import LLaMoDM

from pipeline.trainer.stage import LLaMoStage
import torch.multiprocessing as mp
from pipeline.config import load_config
from llamo.configuration_llamo import GraphConfig

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_scatter")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_sparse")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

# torch.set_default_dtype(torch.float16)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)

class MyDDPStrategy(strategies.DDPStrategy):
    def load_model_state_dict(self, checkpoint):
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)

def main(args):
    mp.set_sharing_strategy('file_system')
    pl.seed_everything(args.seed)

    config = load_config(args.config_file)
    config = GraphConfig.from_dict(config)

    if args.mode == 'eval' and args.stage_path:
        model = LLaMoStage(args, config)
        model.mllm.init_peft()
        ckpt = torch.load(args.stage_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=True)
        print(f"loaded model from {args.stage_path}")

    elif args.stage_path:
        model = LLaMoStage(args, config)
        ckpt = torch.load(args.stage_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=True)
        if config.llm_tune == 'lora':
            model.mllm.init_peft()
        print(f"loaded model from {args.stage_path}")

    else:
        model = LLaMoStage(args, config)

    print('model:', model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    tokenizer = model.tokenizer
    
    # data
    dm = LLaMoDM(args.mode, args.num_workers, args.batch_size, args.root_train, args.root_eval, config.text_max_len, tokenizer, args, config)

    callbacks = []
    ## fixme save only used parameters
    
    callbacks.append(plc.ModelCheckpoint(dirpath="./all_checkpoints/"+args.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs, 
                                         save_last=True, 
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))
    if len(args.devices.split(',')) > 1:
        if args.strategy_name == 'fsdp':
            strategy = strategies.DDPFullyShardedNativeStrategy()
        elif args.strategy_name == 'deepspeed':
            strategy = strategies.DeepSpeedStrategy(stage=3)
        else:
            strategy = MyDDPStrategy(find_unused_parameters=True, start_method='spawn')
    else:
        strategy = 'auto'
        args.devices = eval(args.devices)
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    # trainer = Trainer.from_argparse_args(args,
    #                                      callbacks=callbacks,
    #                                      strategy=strategy,
    #                                      logger=logger,
    #                                     #  limit_train_batches=100,
    #                                      )
    trainer = Trainer(accelerator=args.accelerator, devices=args.devices, precision=args.precision, max_epochs=args.max_epochs, check_val_every_n_epoch=args.check_val_every_n_epoch, callbacks=callbacks, strategy=strategy, logger=logger)
    if args.mode in {'train'}:
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='test.yaml')
    parser.add_argument('--filename', type=str, default="test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy_name', type=str, default=None)
    parser.add_argument('--iupac_prediction', action='store_true', default=False)
    parser.add_argument('--ckpt_path', type=str, default=None)

    # parser = Trainer.add_argparse_args(parser)
    parser = LLaMoStage.add_model_specific_args(parser)
    parser = LLaMoDM.add_model_specific_args(parser)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    #######
    parser.add_argument('--project', type=str, default='linear')
    parser.add_argument('--mlp_depth', type=int, default=2)
    
    
    
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args

if __name__ == '__main__':
    main(get_args())

