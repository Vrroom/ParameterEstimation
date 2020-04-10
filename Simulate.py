from Util import *
from scipy.integrate import odeint

def simulator (model, x0, T) :
    dx = model.dx
#    with stdout_redirected() : 
    result = odeint(dx, x0, T)
    return result

