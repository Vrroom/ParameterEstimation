from Model import SpaxireAgeStratified
from itertools import product
from Simulate import odeSimulator
import math
from Losses import squaredLoss
from Data import Data
import numpy as np
from Date import DateIter, Date
import json

def betaSearch (data, place, betaRange, ldRange) :
    global testingRates
    f = 0.2; numBins = 3
    
    P = data.placeData[place]['P']
    index = 0
    while P[index] == 0:
        index += 1
        if index == len(P):
            # print("No Cases")
            return -1, -1
    P = P[index:]
    T = list(data.placeData[place]['dates'])[index:]
    T = DateIter(Date(T[0]), Date(T[-1])).toList()
    startingTested = P[0]
    
    # print(place, "| Starting:", data.placeData[place]['dates'][index], "| Initial Positive:", startingTested)
    totalInfected = startingTested * 13
    P0 = startingTested; A0 = totalInfected * (1.0 - f) - P0 ;I0 = 0;
    Xs0 = 0;Xa0 = 0;Xi0 = 0; R0 = 0; Xe0 = 0; E0 = f * totalInfected
    
    placeIdx = data.places.index(place)
    Nbar = data.ageBins3[placeIdx]

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
    x0 = np.array(Sbar0 + Ebar0 + Abar0 + Ibar0 + Xsbar0 + Xebar0 + Xabar0 + Xibar0 + Pbar0 + Rbar0)
    model = SpaxireAgeStratified(data, place)
    model.setTestingFractions(*testingRates[place][-3:])
    model.contactTotal = lambda t: data.contactHome + data.contactWork + data.contactOther # no scool in contact matrix

    minLoss, minBeta, minLd = math.inf, None, None

    for b, ld in product(betaRange, ldRange) : 
        model.setBeta(b, ld)
        result = odeSimulator(model, x0, T)
        loss = squaredLoss(result[:, -5], P)
        if loss < minLoss:
            minLoss, minBeta, minLd = loss, b, ld
        # print(result[:, -5], P)
    return minBeta, minLd

class Estimate():
    def __init__(self, beta, ld):
        self.beta = beta
        self.ld = ld

    def isBetaValid(self):
        return self.beta > 0.

    def isLdValid(self):
        return self.ld > 0. and self.ld < 1.

    def print(self, state):
        global testingRates
        # print(','.join([state, str(self.beta), str(self.ld)]))
        print(','.join([state, str(self.beta), str(self.ld)] + [str(x) for x in testingRates[state][-3:]]))

def refineEstimates(estimates):
    validBetas = [x.beta for _, x in estimates.items() if x.isBetaValid()]
    averageBeta = sum(validBetas) / len(validBetas)
    validLds = [x.ld for _, x in estimates.items() if x.isLdValid()]
    averageLd = sum(validLds) / len(validLds)
    for state in estimates.keys():
        if not estimates[state].isBetaValid():
            estimates[state].beta = averageBeta
        if not estimates[state].isLdValid():
            estimates[state].ld = averageLd
    return estimates

if __name__ == "__main__":
    global testingRates
    data = Data('./config.json')
    testingRates = json.load(open('Data/beta.json'))
    betaRange = np.linspace(0, 0.1, 21)
    ldRange = np.linspace(0, 1, 21)
    estimates = {}
    for state in data.places:
        beta, ld = betaSearch(data, state, betaRange, ldRange)
        estimates[state] = Estimate(beta, ld)
    estimates = refineEstimates(estimates)
    for state in data.places:
        estimates[state].print(state)
    
