import torch
from tqdm import tqdm
from torch.autograd import Variable
from functools import partial
from Util import *
from Model import *
import numpy as np
import matplotlib.pyplot as plt

def getProcessJacobians(f, x):
    w = torch.zeros(x.shape)
    w.requires_grad = True
    x.requires_grad = True
    aGrads = []
    wGrads = []
    out = f(x + w)
    for i, _ in enumerate(x) : 
        out[i].backward(retain_graph=True)
        aGrads.append(x.grad.data.clone())
        wGrads.append(w.grad.data.clone())
        x.grad.data.zero_()
        w.grad.data.zero_()
    A = torch.stack(aGrads).numpy()
    W = torch.stack(wGrads).numpy()
    return A, W

def dummyKF (updateStep, x0, P0, Q, H, R, Z, interval) : 
    xPrev = x0
    PPrev = P0
    xs = [x0]
    Ps = [P0]
    for i in tqdm(interval) :
        xPrev = updateStep(xPrev, i)
        PPrev = PPrev
        xs.append(xPrev)
        Ps.append(PPrev)
    return np.stack(xs), Ps

def extendedKalmanFilter (updateStep, x0, P0, Q, H, R, Z, interval) :
    """
    All H, R, z are functions of the date. 
    """
    xPrev = x0
    PPrev = P0
    xs = [x0]
    Ps = [P0]
        
    for i in tqdm(interval) :
        # Time update
        xtMinus = updateStep(xPrev, i)
        A, W = getProcessJacobians(partial(updateStep, t=i, module=torch), torch.from_numpy(xPrev))
        PMinus = A @ PPrev @ A.T + W @ Q @ W.T

        z = Z(i)
        # If there is an observation, 
        # use it to do a measurement update.
        # Else, propagate the time update
        # values.
        if z.size > 0 : 
            # Measurement update
            h = H(i)
            r = R(i)
            K = PMinus @ h.T @ np.linalg.inv(h @ PMinus @ h.T + r)
            xt = xtMinus + K @ (z - h @ xtMinus)
            Pt = (np.eye(PPrev.shape[0]) - K @ h) @ PMinus

            # Shameless hack to ensure values
            # don't go below 0.
            xt[xt < 0] = np.maximum(0, xPrev[xt < 0])

            xPrev = xt
            PPrev = Pt
        else : 
            xPrev = xtMinus
            PPrev = PMinus
        
        xs.append(xPrev)
        Ps.append(PPrev)

    return np.stack(xs), Ps

