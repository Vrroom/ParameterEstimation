import numpy as np 

def squaredLoss(preds, target):
    loss = ((preds - target) ** 2).sum()
    return loss

def squaredLossExpScale(preds, target):
    eps = 1e-8
    preds = np.log(preds + eps)
    target = np.log(target + eps)
    loss = ((preds - target) ** 2).sum()
    return loss

