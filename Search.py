from itertools import product
from functools import partial
from Simulate import *
from Model import * 
from Search import *
import math

def gridSearch (ivRanges, paramRanges, groundTruth, lossFunction, T) :
    lo, hi = T.min(), T.max()
    samples = 5
    T_ = np.linspace(lo, hi, (hi - lo) * samples)
    startIdx = groundTruth[groundTruth['Date'] == '20 Mar'].index[0] 
    deaths = groundTruth['New Deaths'][startIdx:].to_numpy()
    nDays = deaths.size
    minLoss = math.inf
    minx0, minParams = None, None
    for x0, params in product(product(*ivRanges), dictProduct(paramRanges)) :
        model = Sixer(x0, params)
        result = simulator(model, T_)
        infections = result[:, 2][::samples]
        tested = result[:, -1][::samples]
        deathEstimate = 0.02 * (infections + tested)
        deathEstimate = deathEstimate[:nDays]
        loss = lossFunction(deaths, deathEstimate)
        if loss < minLoss : 
            minx0 = x0
            minParams = params
            minLoss = loss

    return minx0, minParams

