# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
# PyTorch library
#
import torch


# MAPE Loss function
#
def MAPELoss(output, target):
    return torch.mean(torch.abs((target - output) / target))    

# SMAPE Loss function
#
def SMAPELoss(output, target):
    return torch.mean(torch.abs(target - output) / ((torch.abs(target) + torch.abs(output)) / 2))   

# def CustomLoss(output, target):
#     return( (torch.cosine_similarity(output, target)**2).mean() )