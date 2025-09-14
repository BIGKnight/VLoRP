import math
import warnings
from typing import Callable, Iterable, Tuple
import torch
import os
from torch import nn
from torch.optim import Optimizer
from transformers.utils.versions import require_version
from transformers.pytorch_utils import Conv1D
from peft_pretraining import training_utils


class ProjFactor:
    def __init__(
        self,
        model,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        rank: int = 32,
        update_proj_gap: int = 100,
        scale: float = 1.0, 
        factor: float = 1.0,
        scheduler: str ="linear",
        gradient_accumulation: int = 1,    
        num_training_steps: int = 10000,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        proj_matrix_dist: str = "normal",
    ):
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        
        assert gradient_accumulation > 0, "gradient_accumulation should be greater than 0"
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.factor = factor
        self.model = model
        self.num_training_steps = num_training_steps
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio

        self.init_lr = lr
        self.cur_lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.correct_bias = correct_bias
        self.gradient_accumulation = gradient_accumulation
        self.proj_matrix_dist = proj_matrix_dist
        
        self.optimizer_dict, self.scheduler_dict, self.id_rp_params = self._layer_wise_initialize()
        self._register_optimizer_hook()

    def _layer_wise_initialize(self):
        rp_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear) and not isinstance(module, Conv1D):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                print('skip module: ', module_name)
                continue
            
            rp_params.append(module.weight)
        id_rp_params = [id(p) for p in rp_params]
        print(f"Total params enabled: {sum(p.numel() for p in rp_params) / 1_000_000:.2f}M")
        optimizer_dict = {}
        scheduler_dict = {}
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        for p in trainable_params:
            if id(p) in id_rp_params:
                assert len(p.shape) == 2, "rp params should be 2D"
                optimizer_dict[p] = SingleParameterRandomAdaFactor(
                    [p],
                    lr=self.init_lr, 
                    weight_decay=self.weight_decay, 
                    betas=self.betas, 
                    eps=self.eps, 
                    correct_bias=self.correct_bias, 
                    rank=self.rank,
                    update_proj_gap=self.update_proj_gap,
                    scale=self.scale,
                    factor=self.factor,
                    gradient_accumulation=self.gradient_accumulation,
                    proj_matrix_dist=self.proj_matrix_dist
                )
            else:
                optimizer_dict[p] = SingleParameterAdamW(
                    [p], 
                    lr=self.init_lr, 
                    weight_decay=self.weight_decay, 
                    betas=self.betas, 
                    eps=self.eps, 
                    gradient_accumulation=self.gradient_accumulation
                ) 
            scheduler_dict[p] = training_utils.get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=self.scheduler,
                    num_training_steps=self.num_training_steps,
                    warmup_steps=self.warmup_steps,
                    min_lr_ratio=self.min_lr_ratio,
                )

                
        return optimizer_dict, scheduler_dict, id_rp_params
    
    def calculate_grad_size(self):
        grad_size = 0
        for p in self.model.parameters():
            if p.requires_grad:
                if id(p) in self.id_rp_params:
                    grad_size += self.optimizer_dict[p].proj_grad.numel()
                else:
                    grad_size += p.numel()
        return grad_size
        
    def _register_optimizer_hook(self):
        for param in self.model.parameters():
            if param.requires_grad:
                def _post_accumulate_grad_hook(p): 
                    if p.grad is None: return
                    elif id(p) in self.id_rp_params: 
                        proj_grad = self.optimizer_dict[p].project(p.grad)

                        if self.optimizer_dict[p].proj_grad is None:
                            self.optimizer_dict[p].proj_grad = proj_grad
                        else:
                            self.optimizer_dict[p].proj_grad += proj_grad
                        p.grad = None # clear grad after accumulating

                    if self.optimizer_dict[p].update_signal():
                        self.optimizer_dict[p].step()
                        self.scheduler_dict[p].step()
                        self.optimizer_dict[p].zero_grad() # for rp params, clear proj_grad by the re-write function of zero_grad
                        # self.accumulate_count = 0 # reset accumulate count
                        self.cur_lr = self.optimizer_dict[p].param_groups[0]["lr"]
                
                param.register_post_accumulate_grad_hook(_post_accumulate_grad_hook)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        if closure is not None: warnings.warn("Closure is not supported in ProjFactor")
        pass

    @torch.no_grad()
    def zero_grad(self):
        pass

class SingleParameterAdamW(torch.optim.AdamW):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        gradient_accumulation: int = 1
    ):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.accumulate_count = 0
        self.gradient_accumulation = gradient_accumulation

    def update_signal(self):
        self.accumulate_count += 1
        return (self.accumulate_count % self.gradient_accumulation) == 0


class SingleParameterRandomAdaFactor(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        rank: int = 32,
        update_proj_gap: int = 100,
        scale: float = 1.0,
        factor: float = 1.0,
        gradient_accumulation: int = 1,
        proj_matrix_dist: str = "normal",
    ):
        assert len(params) == 1, "only support single param"
        assert len(params[0].shape) == 2, "params should be 2D, get shape {}".format(params[0].shape)

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)
        self.proj_grad = None
        self.proj_grad_float64 = None
        self.random_seed = torch.randint(0, 100000, (1,)).numpy().item()
        self.multiplier_pos = None
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.factor = factor
        self.params_shape = params[0].shape
        self.r_num, self.c_num = self.params_shape
        self.update_step = 0
        self.accumulate_count = 0
        self.gradient_accumulation = gradient_accumulation
        self.proj_matrix_dist = proj_matrix_dist

        if self.r_num >= self.c_num:
            assert (self.r_num / self.factor).is_integer() and (self.c_num * self.factor).is_integer()
            self.multiplier_pos = "right"
        else:
            assert (self.r_num * self.factor).is_integer() and (self.c_num / self.factor).is_integer()
            self.multiplier_pos = "left"

    def update_signal(self):
        self.accumulate_count += 1
        return (self.accumulate_count % self.gradient_accumulation) == 0

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        p = group["params"][0]
        assert len(self.param_groups) == 1, "only support single param"
        assert len(group["params"]) == 1, "only support single param"
        assert len(self.proj_grad.shape) == 2, "grad should be 2D"
        if self.proj_grad is None: return loss
        state = self.state[p]

        self.update_step += 1
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(self.proj_grad)
            state["exp_avg_sq_row"] = torch.zeros(self.r_num, device=self.proj_grad.device)
            state["exp_avg_sq_col"] = torch.zeros(self.c_num, device=self.proj_grad.device)
        exp_avg = state["exp_avg"]
        exp_avg_sq_row, exp_avg_sq_col = state["exp_avg_sq_row"], state["exp_avg_sq_col"]
        beta1, beta2 = group["betas"]
        exp_avg.mul_(beta1).add_(self.proj_grad, alpha=(1.0 - beta1))
        proj_back_grad = self.project_back(self.proj_grad)
        exp_avg_sq_row.mul_(beta2).add_((proj_back_grad ** 2).sum(dim=1), alpha=1.0 - beta2)
        exp_avg_sq_col.mul_(beta2).add_((proj_back_grad ** 2).sum(dim=0), alpha=1.0 - beta2)
        step_size = group["lr"]
        if group["correct_bias"]:  # No bias correction for Bert
            bias_correction1 = 1.0 - beta1 ** self.update_step
            bias_correction2 = 1.0 - beta2 ** self.update_step
            step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
        approx_V = torch.mul(exp_avg_sq_row.reshape(-1, 1), exp_avg_sq_col.reshape(1, -1)) / (exp_avg_sq_row.sum())
        norm_grad = self.project_back(exp_avg) / approx_V.sqrt().add_(group["eps"])
        p.add_(norm_grad, alpha=-step_size)
        if group["weight_decay"] > 0.0:
            p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
        return loss
        
    def zero_grad(self) -> None:
        r"""Resets the gradients of all optimized :class:`torch.Tensor` s."""
        self.proj_grad = None
        self.proj_grad_float64 = None

    def project(self, full_rank_grad):
        assert self.r_num == full_rank_grad.shape[0] and self.c_num == full_rank_grad.shape[1],  f"shape mismatch: {self.r_num} vs {full_rank_grad.shape[0]}, {self.c_num} vs {full_rank_grad.shape[1]}"
        if self.update_step % self.update_proj_gap == 0: self.random_seed = torch.randint(0, 100000, (1,)).numpy().item()
        g = torch.Generator(device=full_rank_grad.device).manual_seed(self.random_seed)
        if self.multiplier_pos == "right":
            if self.proj_matrix_dist == "normal":
                random_matrix = torch.randn(
                    self.rank, int(self.c_num / self.factor), 
                    generator=g, 
                    device=full_rank_grad.device, 
                    dtype=full_rank_grad.dtype
                )
            elif self.proj_matrix_dist == "rademacher":
                random_matrix = torch.randint(
                    0, 2, 
                    (self.rank, int(self.c_num / self.factor)), 
                    generator=g, 
                    device=full_rank_grad.device, 
                    dtype=full_rank_grad.dtype
                ) * 2 - 1
            low_rank_grad = torch.matmul(full_rank_grad.reshape(int(self.r_num * self.factor), int(self.c_num / self.factor)), random_matrix.t())
        else:
            if self.proj_matrix_dist == "normal":
                random_matrix = torch.randn(
                    int(self.r_num / self.factor), self.rank, 
                    generator=g, 
                    device=full_rank_grad.device, 
                    dtype=full_rank_grad.dtype
                )
            elif self.proj_matrix_dist == "rademacher":
                random_matrix = torch.randint(
                    0, 2, 
                    (int(self.r_num / self.factor), self.rank), 
                    generator=g, 
                    device=full_rank_grad.device, 
                    dtype=full_rank_grad.dtype
                ) * 2 - 1
            low_rank_grad = torch.matmul(random_matrix.t(), full_rank_grad.reshape(int(self.r_num / self.factor), int(self.c_num * self.factor)))
        del random_matrix # save memory
        return low_rank_grad
    
    def project_back(self, low_rank_grad):
        g = torch.Generator(device=low_rank_grad.device).manual_seed(self.random_seed)
        if self.multiplier_pos == "right":
            if self.proj_matrix_dist == "normal":
                random_matrix = torch.randn(
                    self.rank, int(self.c_num / self.factor), 
                    generator=g, 
                    device=low_rank_grad.device, 
                    dtype=low_rank_grad.dtype
                )
            elif self.proj_matrix_dist == "rademacher":
                random_matrix = torch.randint(
                    0, 2, 
                    (self.rank, int(self.c_num / self.factor)), 
                    generator=g, 
                    device=low_rank_grad.device, 
                    dtype=low_rank_grad.dtype
                ) * 2 - 1
            full_rank_grad = (torch.matmul(low_rank_grad, random_matrix) / self.rank).reshape(self.r_num, self.c_num)
        elif self.multiplier_pos == "left":
            if self.proj_matrix_dist == "normal":
                random_matrix = torch.randn(
                    int(self.r_num / self.factor), self.rank, 
                    generator=g, 
                    device=low_rank_grad.device, 
                    dtype=low_rank_grad.dtype
                )
            elif self.proj_matrix_dist == "rademacher":
                random_matrix = torch.randint(
                    0, 2, 
                    (int(self.r_num / self.factor), self.rank), 
                    generator=g, 
                    device=low_rank_grad.device, 
                    dtype=low_rank_grad.dtype
                ) * 2 - 1
            full_rank_grad = (torch.matmul(random_matrix, low_rank_grad) / self.rank).reshape(self.r_num, self.c_num)
        del random_matrix # save memory
        return full_rank_grad * self.scale