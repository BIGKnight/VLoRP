import torch
import transformers
from peft_pretraining import training_utils
from transformers.pytorch_utils import Conv1D
import torch.nn as nn
from optimizers import ProjFactor

def optimizer_constructor(args, logger, model):
    if args.optimizer.lower() == "projfactor":
        optimizer = ProjFactor( 
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            rank=args.rank,
            update_proj_gap=args.update_proj_gap,
            scale=args.scale,
            factor=args.factor,
            scheduler=args.scheduler,
            gradient_accumulation=args.gradient_accumulation,
            warmup_steps=args.warmup_steps,
            num_training_steps=args.num_training_steps,
            min_lr_ratio=args.min_lr_ratio,
            correct_bias=True,
            proj_matrix_dist=args.proj_matrix_dist,
        )
        return optimizer, None