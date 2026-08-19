import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()   # 對應 Java 的 super()
        # 接下來建立你的權重參數...
        weight_tensor = torch.empty((out_features, in_features), device=device, dtype=dtype)#前面是形狀
        std = (2 / (in_features + out_features)) ** 0.5 #標準差
        torch.nn.init.trunc_normal_(weight_tensor, mean=0, std=std, a=-3*std, b=3*std)
        self.weight=nn.Parameter(weight_tensor)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return x@self.weight.T