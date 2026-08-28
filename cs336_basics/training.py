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

def gradient_clipping(parameters, max_l2_norm):

    total = 0.0
    eps = 1e-6
    for p in parameters:
        if p.grad is None:
            continue
        total += (p.grad ** 2).sum()
    total_norm = total.sqrt()

    if total_norm>=max_l2_norm:
        scale=max_l2_norm/(total_norm+eps)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad*=scale

import numpy as np
def get_batch(x, batch_size, context_length, device):
    # 隨機選 batch_size 個合法的起始位置
    starts = np.random.randint(0,len(x) - context_length,size=batch_size,)

    inputs = np.stack([x[i : i + context_length]for i in starts])

    targets = np.stack([
        x[i + 1 : i + context_length + 1]for i in starts])

    inputs = torch.from_numpy(inputs.astype(np.int64)).to(device)
    targets = torch.from_numpy(targets.astype(np.int64)).to(device)

    return inputs, targets

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint=torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

import os

def evaluate(model, valid_data, batch_size, context_length, device, num_batches):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = get_batch(valid_data, batch_size, context_length, device)   # 1. 拿一批 validation 資料
            logits = model(inputs)                                                          # 2. 模型做預測(forward)
            logits = logits.reshape(-1, logits.shape[-1])                                    # 3. 攤平成 2 維
            targets = targets.reshape(-1)                                                     # 4. targets 也攤平
            loss = cross_entropy_loss(logits, targets)                                        # 5. 算這一批的 loss
            total_loss += loss.item()                                                          # 6. 累加起來

    model.train()
    return total_loss / num_batches   # 7. 算平均，得到最終的 validation loss

def train(model, train_data, valid_data, optimizer, batch_size, context_length, max_iters, device,
          max_l2_norm, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters,
          checkpoint_path=None, checkpoint_every=100, log_path=None, eval_every=500, eval_batches=20):

    import time
    import csv
    print(device)
    model.to(device)
    model.train()

    # 有 checkpoint 就接著練,沒有就從 0 開始
    start_it = 0
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        start_it = load_checkpoint(checkpoint_path, model, optimizer)

    start_time=time.time()

     # TODO 2: 如果 log_path 有給，開啟這個檔案準備寫入，並寫入表頭
    log_file = None
    log_writer = None   
    if log_path is not None:
        log_file = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["step", "time", "loss", "lr"])

    for it in range(start_it, max_iters):
 # 1. 學習率排程
        lr = get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate,
                                    warmup_iters, cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 2~4. 清梯度 → 取 batch → forward → loss → backward
        optimizer.zero_grad()
        inputs, targets = get_batch(train_data, batch_size, context_length, device)
        logits = model(inputs)
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
        loss = cross_entropy_loss(logits, targets)

        if log_writer is not None:
            elapsed = time.time() - start_time
            log_writer.writerow([it, elapsed, loss.item(), lr])

        loss.backward()

        # 5~6. 裁梯度 → 更新
        gradient_clipping(model.parameters(), max_l2_norm)
        optimizer.step()

        # 定期存檔
        if checkpoint_path is not None and it % checkpoint_every == 0:
            save_checkpoint(model, optimizer, it + 1, checkpoint_path)
        # TODO: 每隔 eval_every 步，呼叫 evaluate，把結果印出來（也可以考慮寫進 log）
        
        if it % eval_every == 0:
            val_loss = evaluate(model, valid_data, batch_size, context_length, device, eval_batches)
            print(f"Iteration {it} | Validation Loss: {val_loss:.4f}")

    # 訓練結束後補存最後一次
    if checkpoint_path is not None:
        save_checkpoint(model, optimizer, max_iters, checkpoint_path)

    if log_file is not None:
        log_file.close()


import numpy as np
import torch
from cs336_basics.model import TransformerLm
from cs336_basics.training import AdamW, train


if __name__ == "__main__":
    train_data = np.load("data/tinystories_train.npy", mmap_mode="r")
    valid_data = np.load("data/tinystories_valid.npy", mmap_mode="r")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_batch_size = 32
    base_lr = 1e-3

    batch_sizes = [1, 4, 16, 32, 64, 128]

    for bs in batch_sizes:
        lr = base_lr * (bs / base_batch_size)

        print(f"=== 開始訓練 batch_size={bs}, lr={lr:.6f} ===")

        model = TransformerLm(
            vocab_size=10000, context_length=256, d_model=512,
            num_layers=4, num_heads=16, d_ff=1344, rope_theta=10000.0,
        )

        optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        # train(
        #     model, train_data, valid_data, optimizer,
        #     batch_size=bs, context_length=256, max_iters=1000,
        #     device=device, max_l2_norm=1.0,
        #     max_learning_rate=lr, min_learning_rate=lr * 0.1,
        #     warmup_iters=50, cosine_cycle_iters=1000,
        #     checkpoint_path=None,
        #     log_path=f"logs/batch_size_{bs}.csv",
        #     eval_every=200,
        #     eval_batches=10,
        # )
        
        import csv
    import matplotlib.pyplot as plt

    batch_sizes = [1, 4, 16, 32, 64, 128]

    # ---- 圖 1: Training Loss(從 CSV 讀取)----
    plt.figure(figsize=(10, 6))
    for bs in batch_sizes:
        steps = []
        losses = []
        with open(f"logs/batch_size_{bs}.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)   # 跳過表頭
            for row in reader:
                steps.append(int(row[0]))
                losses.append(float(row[2]))
        plt.plot(steps, losses, label=f"batch_size={bs}")

    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curves for Different Batch Sizes")
    plt.legend()
    plt.savefig("logs/batch_size_training_loss.png")
    plt.show()

    # ---- 圖 2: Validation Loss(用你終端機貼出來的數字)----
    validation_data = {
        1:   [(0, 9.2704), (200, 6.2015), (400, 5.3630), (600, 5.1051), (800, 4.9333)],
        4:   [(0, 9.2759), (200, 4.3963), (400, 3.9741), (600, 3.6324), (800, 3.4448)],
        16:  [(0, 9.2484), (200, 3.3077), (400, 2.8905), (600, 2.6888), (800, 2.5501)],
        32:  [(0, 9.2654), (200, 2.9433), (400, 2.5371), (600, 2.3610), (800, 2.2126)],
        64:  [(0, 9.2286), (200, 2.6333), (400, 2.2435), (600, 2.0334), (800, 1.9290)],
        128: [(0, 9.2899), (200, 2.5460), (400, 2.2211), (600, 2.0262), (800, 1.9182)],
    }

    plt.figure(figsize=(10, 6))
    for bs, points in validation_data.items():
        steps = [p[0] for p in points]
        losses = [p[1] for p in points]
        plt.plot(steps, losses, marker="o", label=f"batch_size={bs}")

    plt.xlabel("Step")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Curves for Different Batch Sizes")
    plt.legend()
    plt.savefig("logs/batch_size_validation_loss.png")
    plt.show()
    