import numpy as np 

def heteroscedastic_loss(true, mean, var):
    precision = 1/var
    log_var= torch.log(var)    
    return torch.mean(torch.sum(precision * (true - mean)**2 + log_var, 1), 0)

def squaredLoss(preds, target):
    loss = ((preds - target) ** 2).sum()
    return loss

def squaredLossExpScale(preds, target):
    eps = 1e-8
    preds = np.log(preds + eps)
    target = np.log(target + eps)
    loss = ((preds - target) ** 2).sum()
    return loss

