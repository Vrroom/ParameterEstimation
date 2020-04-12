from Util import *
from scipy.integrate import odeint
from functools import partial
import numpy as np 

def simulator (model, x0, T) :
    dx = partial(model.dx, module=np)
#    with stdout_redirected() : 
    result = odeint(dx, x0, T)
    return result

