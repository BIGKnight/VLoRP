import os
import time
import json
import random
import argparse
import numpy as np

from requests import get
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.modeling_llama import LlamaForCausalLM
from Utils import optimizer_constructor, model_constructor, dataloader_constructor, get_lora_model

transformers.logging.set_verbosity_error() 

def parse_args(args):
    parser = argparse.ArgumentParser()

    # parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)   
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)
    
    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")
    parser.add_argument("--proj_mode", type=str, default="m1")
    parser.add_argument("--proj_matrix_dist", type=str, default="normal", choices=["gaussian", "rademacher"])
    parser.add_argument('--wandb_expname', type=str, default='c4_ft')
    parser.add_argument(
        "--factor",
        type=float,
        default=1.,
    )
    
    # LoRA parameters
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    parser.add_argument("--train_on_prompts", default=False, action="store_true")
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")
    args = parser.parse_args(args)
    args = args_utils.check_args_torchrun_main(args)
    print(args.save_dir)
    return args


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()
            
    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        wandb.init(project="projfactor", name=args.wandb_expname)
        
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)
    model, tokenizer = model_constructor(args.model_name, args.max_length)
    
    lora_or_not = "lora_" in args.optimizer.lower()
    if lora_or_not:
        args.optimizer = args.optimizer.lower().replace("lora_", "")
        model = get_lora_model(model, rank=args.rank, lora_alpha=args.lora_alpha, target_modules="all-linear")

    print(f"Saving model to {args.save_dir} every {args.save_every} update steps")
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    dataset, dataloader = dataloader_constructor(
        args.dataset, 
        args.batch_size, 
        args.max_length, tokenizer, 
        args.workers, args.train_on_prompts,
    )

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": args.dataset,
        "model": args.model_name,
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)
    
    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    optimizer, scheduler = optimizer_constructor(args, logger, model)

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # TRAINING LOOP
    # ##############################
    optimizer_type_flag = isinstance(optimizer, torch.optim.Optimizer)
    log_lr = optimizer.param_groups[0]["lr"] if optimizer_type_flag else optimizer.cur_lr
    for batch_idx, batch in enumerate(dataloader):
        global_step += 1
        local_step += 1
        if update_step >= args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        model.train()
        batch = {k: v.to(device) for k, v in batch.items()}
        # show the memory
        if not ("labels" in batch.keys()):
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
            tokens_seen += (batch["input_ids"]).sum().item() * world_size
            res = model(**batch, labels=labels)
        else:
            tokens_seen += (batch["input_ids"]).sum().item() * world_size
            res = model(input_ids=batch["input_ids"], labels=batch["labels"], attention_mask=batch["attention_mask"])
        
        loss = res.loss
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()
        if global_step % args.gradient_accumulation != 0:
            continue

        # The below code is only executed during the update step
        # add grad clipping
        if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
        if global_rank == 0: pbar.update(1)

        optimizer.step()
        if scheduler is not None: # for layer-wise updates, scheduler is set internally in optimizer, so no need to call scheduler.step()
            scheduler.step()
        optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time

        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            wandb.log({
                "loss": loss.item(),
                "lr": log_lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                },
                step=update_step,
            )
        optimizer_type_flag = isinstance(optimizer, torch.optim.Optimizer)
        log_lr = optimizer.param_groups[0]["lr"] if optimizer_type_flag else optimizer.cur_lr
        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    if lora_or_not: 
        logger.info("merging and unloading lora model")
        model = model.merge_and_unload()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
    os.makedirs(args.save_dir, exist_ok=True)
    if args.single_gpu:
        model.save_pretrained(current_model_directory, max_shard_size='100GB')
        tokenizer.model_max_length = args.max_length * 100
        tokenizer.save_pretrained(current_model_directory)
    else:
        model.module.save_pretrained(current_model_directory, max_shard_size='100GB')
        tokenizer.model_max_length = args.max_length * 100
        tokenizer.save_pretrained(current_model_directory)
    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")

if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
