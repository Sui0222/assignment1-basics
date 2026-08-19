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

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        embed_weights = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(embed_weights, mean=0, std=1, a=-3, b=3)
        self.weight = nn.Parameter(embed_weights)

    def forward(self,token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
    # d_model: int  Hidden dimension of the model
    # eps: float = 1e-5  Epsilon value for numerical stability
    # device: torch.device | None = None  Device to store the parameters on
    # dtype: torch.dtype | None = None  Data type of the parameters
        self.eps = eps   # eps 之後 forward 會用到,也要存起來
        g_tensor = torch.ones(d_model,device=device, dtype=dtype)
        self.weight=nn.Parameter(g_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Your code here performing RMSNorm
        # TODO 1: 算出 RMS(a) —— 對 x 逐元素平方,沿最後一維取平均,加 eps,開根號
        rms = ((x ** 2).mean(dim=-1, keepdim=True) + self.eps) ** 0.5
        # TODO 2: 用 x 除以 rms,再乘上 self.weight,得到最終結果
        result = x/rms*self.weight
        return result.to(in_dtype)