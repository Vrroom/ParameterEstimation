import torch
from tqdm import tqdm
from torch.autograd import Variable
from functools import partial
from Util import *
from Model import *
from Simulate import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.integrate
import pdb
import numdifftools as nd

def getProcessJacobians(f, x):
    n = np.size(x)
    eta = 1e-2
    for i in range(0,n):
        dx = np.zeros(n)
        dx[i] = eta
        df = (f(x + dx) - f(x))/eta

        if i == 0:
            jac = df
        else:
            jac = np.vstack((jac,df))
    return jac.T

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

    d1 = Date('27 Apr')
    #print(tEnd.date)
        
    for date in tqdm(DateIter(tStart, tEnd)) :
        # Time update
        i = date - tStart
        #pdb.set_trace()
        xtMinus = scipy.integrate.odeint(updateStep,xPrev,[i,i+1],args=(tStart,))
        #pdb.set_trace()
        xtMinus = xtMinus[1]
        A = getProcessJacobians(partial(updateStep, delta_t=i, startDate=tStart), xPrev)
        #A = nd.Jacobian(partial(updateStep, delta_t=i, startDate=tStart))(xPrev)
        phi = scipy.linalg.expm(A)
        PMinus = phi @ PPrev @ phi.T + Q

        # Measurement update
        h = H(date+1)
        r = R(date+1)
        z = Z(date+1)

        #if(date >= d1):
        	#pdb.set_trace()

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

        #print(date.date)

    return np.stack(xs), Ps

if __name__ == "__main__" : 
    print(getJacobian(sin, torch.tensor([0., 1.])))
