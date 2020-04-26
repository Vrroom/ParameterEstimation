import more_itertools
import contextlib
import math
import pandas
import sys
import numpy as np
from itertools import product
import datetime
import os
import os.path as osp

def sortAndFlattenDict(d) : 
    return list(unzip(sorted(d.items()))[1])

def dictProduct (d) : 
    return map(dict, product(*map(lambda x : product([x[0]], x[1]), d.items())))

def climbFn (t, ti, tf, xi, xf) : 
    if t > tf : 
        return xf
    elif ti < t < tf : 
        wt = (t - ti) / (tf - ti)
        return xf * wt + xi * (1 - wt)
    else : 
        return xi

def stepFn (t, t0, x1, x2) : 
    if t > t0 : 
        return x2
    else :
        return x1

def bumpFn (t, ti, tf, x1, x2) : 
    if t < ti or t > tf : 
        return x1
    else :
        return x2

def sigmoid (x) : 
    return 1 / (1 + math.e ** -x)

def transpose (a) : 
    nCol = len(a[0])
    return [[r[i] for r in a] for i in range(nCol)] 

