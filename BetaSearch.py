from Model import SpaxireAgeStratified
from itertools import product
from Simulate import odeSimulator
import math
from Losses import squaredLoss

def betaSearch (data, place, x0, betaRange, ldRange) :
    # TODO : Maybe fixing initialization 
    # will fix beta values
    totalInfected = startingTested * 20
    P0 = startingTested; A0 = totalInfected * (1.0 - f) - P0 ;I0 = 0;
    Xs0 = 0;Xa0 = 0;Xi0 = 0; R0 = 0; Xe0 = 0; E0 = f * totalInfected
    
    Pbar0 = [0, P0, 0]
    Rbar0 = [0, 0, 0]
    Abar0 = [0, A0, 0]
    Ibar0 = [0, I0, 0]
    Ebar0 = [0, E0, 0]
    Xabar0 = [0, 0, 0]
    Xibar0 = [0, 0, 0]
    Xebar0 = [0, 0, 0]
    Xsbar0 = [0, 0, 0]
    Sbar0 = [Nbar[i] - Abar0[i] - Ibar0[i] - Ebar0[i] - Xsbar0[i] - Xabar0[i] - Xibar0[i] - Xebar0[i] - Pbar0[i] - Rbar0[i] for i in range(numBins)]
    # TODO : Move x0 from argument to inside.
    model = SpaxireAgeStratified(data, place)
    P = data.placeData[place]['P']
    T = np.arange(P.size)
    minLoss, minBeta, minLd = math.inf, None, None

    for b, ld in product(betaRange, ldRange) : 
        model.setBeta(b, ld)
        result = odeSimulator(model, x0, T)
        loss = squaredLoss(result[:, -2], P)
        if loss < minLoss
            minLoss, minBeta, minLd = loss, b, ld

    return b, ld

    
