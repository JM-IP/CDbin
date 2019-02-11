
# coding: utf-8

# In[8]:


import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
class CorrelationPenaltyLoss(nn.Module):
    def __init__(self):
        super(CorrelationPenaltyLoss, self).__init__()

    def forward(self, input):
        mean1 = torch.mean(input, dim=0)
        zeroed = input - mean1.expand_as(input)
        cor_mat = torch.bmm(torch.t(zeroed).unsqueeze(0), zeroed.unsqueeze(0)).squeeze(0)
        # bmm stand for batch matrix multiply
        # t stands for transform
        # unsqueeze(0) stands for adding a dim as 1 to the matrix
        d = torch.diag(torch.diag(cor_mat))
        no_diag = cor_mat - d
        d_sq = no_diag * no_diag
        return torch.sqrt(d_sq.sum())/input.size(0)

class QuantilizeLoss(nn.Module):
    def __init__(self, scale):
        super(QuantilizeLoss, self).__init__()
        self.scale=scale
    def forward(self, input):
        bininput = torch.sign(input)
        bininput = self.scale * bininput
        bininput=bininput.detach()
        return torch.sum(torch.pow(input-bininput,2))/2/input.size(0)/input.size(1)

class Even_distributeLoss(nn.Module):
    def __init__(self):
        super(Even_distributeLoss, self).__init__()
    def forward(self, input):
        mean1 = torch.mean(input, dim=0)
        y = torch.pow(mean1, 2)
        return torch.sum(y)/2/input.size(1)


