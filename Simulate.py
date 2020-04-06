from Util import *
from scipy.integrate import odeint

def simulator (model, T) :
    x0 = sortAndFlattenDict(model.init) 
    dx = sortAndFlattenDict(model.dx)
    return odeint(dx, x0, T)

