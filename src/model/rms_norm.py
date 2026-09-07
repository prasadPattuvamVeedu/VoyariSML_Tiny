import torch
import torch.nn as nn
from  configs.model_config import D_MODEL

class RMSNorm(nn.Module):
    def __init__(self,dim=D_MODEL,eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self,x):

        mean_square = x.pow(2).mean(dim=-1,keepdim=True)
        rms         = torch.sqrt(mean_square+self.eps)
        normalized  = x/rms
        output = normalized * self.weight

        return output