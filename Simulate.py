from Util import *
from scipy.integrate import odeint

def simulator (model, T) :
    x0 = model.init
    dx = model.dx
    return odeint(dx, x0, T)

