# @ 2025 Omni-Mol Project

from loggers import WrappedLogger
from args import ModelArguments, DataArguments, TrainingArguments
import transformers
from dataclasses import asdict
import json
import os
import datetime
from data_pipe.datasets import build_dataset
import train_engine
import torch.distributed as dist
from helpers import model_profiler, seperate_save_lora, seperate_save_partial
import model_factory
import pathlib
import wandb

logger = WrappedLogger(__name__)

def parse_args() -> tuple[ModelArguments, DataArguments, TrainingArguments]:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    return model_args, data_args, training_args


def main(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    # seed everything (Optional)
    transformers.set_seed(0)
    # Dump args
    args = {
        "Model Args": asdict(model_args), 
        "Data Args": asdict(data_args), 
        "Training Args": asdict(training_args)
        }
    if int(os.environ.get("LOCAL_RANK",-1)) in [0, -1]:
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
        with open(os.path.join(training_args.output_dir, "args.json"), mode="w") as f:
            json.dump(args, f, indent=4)
            f.close()
    
    # Create model, tokenizer
    tokenizer, model = model_factory.create_model(model_args, data_args, training_args)
    # create dataset
    data_module = build_dataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        for_test=False,
        add_selfies=data_args.add_selfies,
        split_val=data_args.split_eval,
        val_ratio=data_args.val_ratio,
        task_config=training_args.task_config,
        sample_from_ratio=training_args.sample_from_ratio,
        total_size=training_args.total_size,
        if_smiles=training_args.if_smiles
    )
    
    model_profiler(model, training_args.output_dir)
    
    # save callback
    from transformers import TrainerCallback
    # callback function for model saving
    class SaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            # get saving dir from args
            checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(state.global_step))
            # checkpoint_dir = args.output_dir
            if not 'partial' in training_args.training_recipe:
                seperate_save_lora(args, checkpoint_dir, model)
            else:
                seperate_save_partial(args, checkpoint_dir, model)

            
    class TruncateCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if args.stop_epoch is not None:
                if state.epoch > args.stop_epoch:
                    return {"should_training_stop": True}
    
    # train
    trainer = train_engine.MoETrainer(
        model=model,
        args=training_args,
        callbacks=[SaveCallback(), TruncateCallback()],
        **data_module
    )
    # dist.barrier()
    # If we have saved ckpts, we resume from it and continue training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:  # No savings, train from scratch
        trainer.train()
    
    # Save state dict that is related to training
    trainer.save_state()
    logs = trainer.state.log_history
        
    with open(os.path.join(training_args.output_dir, "logs.json"), mode="w") as f:
        json.dump(logs, f, indent=4)
        f.close()

if __name__ == "__main__":
    logger.info(f"Training script for Omni-Mol", on_rank0=True)
    logger.info(f"Time: \033[34m{datetime.datetime.now()}\033[0m", on_rank0=True)
    model_args, data_args, training_args = parse_args()
    main(model_args, data_args, training_args)

