import torch
from functools import partial
from Util import *
from Model import *
import numpy as np
import matplotlib.pyplot as plt

def getJacobian(f, x):
    x.requires_grad = True
    grads = []
    for i, _ in enumerate(x) : 
        out = f(x)
        out[i].backward()
        grads.append(x.grad.data.clone())
        x.grad.data.zero_()
    return torch.stack(grads).numpy()

def sin(x) :
    if torch.is_tensor(x) : 
        return torch.sin(x)
    else :
        return np.sin(x)

def step(x) :
    A = np.array([[1., 3.], [-1., 2.]])
    if torch.is_tensor(x) : 
        return torch.from_numpy(A) @ x
    else :
        return A @ x

def extendedKalmanFilter (updateStep, x0, P0, H, R, z, tStart, tEnd) :
    xPrev = x0
    PPrev = P0
    xs = [x0]
    Ps = [P0]
        
    for i, date in enumerate(DateIter(tStart + 1, tEnd)) :
        # Time update
        xtMinus = updateStep(xPrev, i+1)
        A = getJacobian(partial(updateStep, t=i+1, module=torch), torch.from_numpy(xPrev))
        PMinus = A @ PPrev @ A.T 
        # Measurement update
        h = H(date)
        K = PMinus @ h.T @ np.linalg.inv(h @ PMinus @ h.T + R)
        xt = xtMinus + K @ (z[i] - h @ xtMinus)
        Pt = (np.eye(PPrev.shape[0]) - K @ h) @ PMinus

        xPrev = xt
        PPrev = Pt
        
        xs.append(xt)
        Ps.append(Pt)

    return np.stack(xs), Ps

if __name__ == "__main__" : 
    x0 = np.array([1., 1.])
    P0 = np.eye(2, 2) * 1e-5
    H = np.array([[1, -1]])
    R = 1
    T = 10

    # Get data
    xs = [x0] 
    for t in range(1, T) :
        xs.append(sin(xs[-1]))

    zs = []
    for x in xs : 
        z = H @ x + np.random.randn()
        zs.append(z)

    xs_ = extendedKalmanFilter(sin, x0 + 1, P0, H, R, zs, T)

    x1 = [r[0] for r in xs]
    y1 = [r[1] for r in xs]

    x2 = [r[0] for r in xs_]
    y2 = [r[1] for r in xs_]

    print(x1)
    print(x2)
    print(y1)
    print(y2)

    plt.plot(range(T), x1, c='violet', label='x real')
    plt.plot(range(T), y1, c='green', label='y real')
    plt.plot(range(T), x2, c='orange', label='x est')
    plt.plot(range(T), y2, c='black', label='y est')

    plt.legend()
    plt.show()
