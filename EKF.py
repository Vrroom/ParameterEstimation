import torch
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
    for i, _ in enumerate(x) : 
        out = f(x + w)
        out[i].backward()
        aGrads.append(x.grad.data.clone())
        wGrads.append(w.grad.data.clone())
        x.grad.data.zero_()
        w.grad.data.zero_()
    A = torch.stack(aGrads).numpy()
    W = torch.stack(wGrads).numpy()
    return A, W

def sin(x) :
    if torch.is_tensor(x) : 
        a, b = x
        return torch.sin(torch.stack([a+b, a-b]))
    else :
        return np.sin(x)

def step(x) :
    A = np.array([[1., 3.], [-1., 2.]])
    if torch.is_tensor(x) : 
        return torch.from_numpy(A) @ x
    else :
        return A @ x

def extendedKalmanFilter (updateStep, x0, P0, Q, H, R, Z, tStart, tEnd) :
    """
    All H, R, z are functions of the date. 
    """
    xPrev = x0
    PPrev = P0
    xs = [x0]
    Ps = [P0]
        
    for i, date in enumerate(DateIter(tStart + 1, tEnd)) :
        # Time update
        xtMinus = updateStep(xPrev, i+1)
        A, W = getProcessJacobians(partial(updateStep, t=i+1, module=torch), torch.from_numpy(xPrev))
        PMinus = A @ PPrev @ A.T + W @ Q @ W.T

        # Measurement update
        h = H(date)
        r = R(date)
        z = Z(date)
        if h.size > 0 : 
            K = PMinus @ h.T @ np.linalg.inv(h @ PMinus @ h.T + r)
            xt = xtMinus + K @ (z - h @ xtMinus)
            Pt = (np.eye(PPrev.shape[0]) - K @ h) @ PMinus

            xt[xt < 0] = np.maximum(xt[xt < 0], np.maximum(0, xPrev[xt < 0])) # Shameless Hack

            xPrev = xt
            PPrev = Pt
        else : 
            xPrev = xtMinus
            PPrev = PMinus

            xt = xtMinus
            Pt = PMinus
        
        xs.append(xt)
        Ps.append(Pt)

    return np.stack(xs), Ps

if __name__ == "__main__" : 
    print(getJacobian(sin, torch.tensor([0., 1.])))
