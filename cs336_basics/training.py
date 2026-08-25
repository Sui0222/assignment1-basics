import torch
import torch.nn as nn

def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch_size = inputs.shape[0]

    max_value, _ = inputs.max(dim=-1, keepdim=True)   # (batch_size, 1)
    max_value = max_value.squeeze(-1)   # (batch_size, 1) → (batch_size,)

    row_indices = torch.arange(batch_size)
    target_logits = inputs[row_indices, targets]        # (batch_size,) —— 這是 x_i

    # TODO 1: 算 log(∑_j e^{x_j - max})，這是對「整個 inputs」做的，不是 target_logits
    #         提示: (inputs - max_value).exp().sum(dim=-1) 再取 log
    log_sum_exp = (inputs-max_value.unsqueeze(-1)).exp().sum(dim=-1)   # 形狀應該是 (batch_size,)
    log_sum_exp = torch.log(log_sum_exp)

    # TODO 2: 組合出每個樣本的 loss
    #         loss_i = -(x_i - max(x) - log_sum_exp)
    #         注意 max_value 形狀是 (batch_size, 1)，這裡要先 squeeze 成 (batch_size,)
    loss = -(target_logits-max_value-log_sum_exp)

    return loss.mean()

from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr,betas,eps,weight_decay,):
        defaults = {"lr": lr,"betas": betas,"eps": eps,"weight_decay": weight_decay,}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # --------------------------------
                # 1. Get / initialize optimizer state
                # --------------------------------

                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                t = state["t"]
                m = state["m"]
                v = state["v"]
                t+=1
                a_t=lr*((1-beta2**t)**0.5/(1-beta1**t))
                p.data-=lr*weight_decay*p.data
                m=beta1*m+(1-beta1)*grad
                v=beta2*v+(1-beta2)*(grad**2)
                p.data-=a_t*m/((v**0.5)+eps)
                state["t"] = t
                state["m"] = m
                state["v"] = v

        return loss

def get_lr_cosine_schedule(it: int,max_learning_rate: float,min_learning_rate: float,warmup_iters: int,cosine_cycle_iters: int,)->float:
    """
    it (int): Iteration number to get learning rate for.
    max_learning_rate (float): alpha_max, the maximum learning rate for
                cosine learning rate schedule (with warmup).
    min_learning_rate (float): alpha_min, the minimum / final learning rate for
                the cosine learning rate schedule (with warmup).
    warmup_iters (int): T_w, the number of iterations to linearly warm-up
                the learning rate.
    cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.
    """
    alpha_t=0
    if it<warmup_iters:
        alpha_t=it/warmup_iters*max_learning_rate
    elif it>=warmup_iters and it <=cosine_cycle_iters:
        alpha_t=min_learning_rate+0.5*(1+math.cos(((it-warmup_iters)/(cosine_cycle_iters-warmup_iters))*math.pi))*(max_learning_rate-min_learning_rate)
    else:
        alpha_t=min_learning_rate
    return alpha_t