from Util import *
import json
import random
from Date import Date
import math
from more_itertools import collapse
from itertools import product
from functools import partial
import torch
import matplotlib.pyplot as plt
import numpy as np
from Simulate import *
from copy import deepcopy
from Plot import * 

cat = {np : np.hstack, torch : torch.cat}

class ConnectedSpaxire () : 

    def __init__ (self, data) : 
        self.data = data
        self.links = [([], []) for _ in data.places]
        self.models = [SpaxireAgeStratified(data, p, *c) for p, c in zip(data.places, self.links)]
        self.nPlaces = len(data.places)
        self.lockdownEnd = Date('3 May')

    def dx (self, x, t, module=np) : 
        xs = x.reshape((self.nPlaces, -1))
        derivatives = [m.dx(x, t, module) for x, m in zip(xs, self.models)]
        dx1 = deepcopy(cat[module](derivatives))
        for m in self.models : 
            m.send()

        for i in range(self.nPlaces) : 
            _, outChannel = self.links[i]
            data = outChannel.pop()
            for j in range(self.nPlaces) :
                data_ = dict()
                for key, val in data.items() : 
                    data_[key] = val * self.data.transportMatrix[j, i]
                inChannel, _ = self.links[j]
                inChannel.append(data_)

        for m in self.models : 
            m.receive()

        derivatives_ = [m.addCrossTerms(dx, module) for m, dx in zip(self.models, derivatives)]
        dx = cat[module](derivatives_)
        print(t.date, dx1.sum(), dx.sum(), x.min(), x.argmin())
        return dx

    def timeUpdate (self, x, t, module=np) : 
        dx = self.dx(x, t, module)
        return x + dx

class SpaxireAgeStratified () : 
    """
    Current ODE Model class. 
    """
    def __init__ (self, data, place, inChannel=None, outChannel=None) :
        self.inChannel = inChannel
        self.outChannel = outChannel

        self.tl = Date('24 Mar') # Lockdown Begin
        self.te = Date('3 May')  # Lockdown End

        self.k0 = partial(bumpFn, ti=self.tl, tf=self.te, x1=0, x2=1/7)
        self.mu = partial(stepFn, t0=self.te, x1=0, x2=1/7)

        self.sigma  = 1 / 5

        self.gamma1 = 1 / 21
        self.gamma2 = 1 / 21
        self.gamma3 = 1 / 21

        self.beta  = 0.015
        self.beta2 = 0.1

        self.f = 0.2
        self.lockdownLeakiness = 0.7

        self.contactHome  = lambda t : data.contactHome
        self.contactTotal = lambda t : data.contactTotal
    
        self.bins = 3 # Age bins
        self.adultBins = [1]

        self.testingFraction1 = lambda t : 1 / 13
        self.testingFraction2 = lambda t : 0
        self.testingFraction3 = lambda t : 0 

        placeIdx = data.places.index(place)
        self.Nbar = deepcopy(data.ageBins3[placeIdx])
        self.mortality = data.mortality[placeIdx]
        self.totalOut = data.transportMatrix[:, placeIdx].sum()

    def send (self) : 
        sOut = self.s[1] / (self.Nbar[1])
        eOut = self.e[1] / (self.Nbar[1]) 
        aOut = self.a[1] / (self.Nbar[1]) 
        iOut = self.i[1] / (self.Nbar[1]) 
        rOut = self.r[1] / (self.Nbar[1]) 

        data = {'s': sOut, 'e': eOut, 'a' : aOut, 'i' : iOut, 'r' : rOut} 
        self.outChannel.append(data)

        self.sOut = self.totalOut * sOut
        self.eOut = self.totalOut * eOut
        self.aOut = self.totalOut * aOut
        self.iOut = self.totalOut * iOut
        self.rOut = self.totalOut * rOut

    def receive (self) : 
        self.sIn = sum([data['s'] for data in self.inChannel])
        self.eIn = sum([data['e'] for data in self.inChannel])
        self.aIn = sum([data['a'] for data in self.inChannel])
        self.iIn = sum([data['i'] for data in self.inChannel])
        self.rIn = sum([data['r'] for data in self.inChannel])
        self.inChannel.clear()

    def dx (self, x, t, module=np) : 
        """
        This gives the derivative wrt time
        of the state vector. 

        This function can be directly plugged
        into scipy's odeint with the initial 
        values to simulate the model.

        Parameters
        ----------
        x : state vector
        t : time step 
        module : whether to use torch or numpy
        """
        s, e, a, i, xs, xe, xa, xi, p, r = x.reshape((-1, self.bins))
    
        # convert depending on usage of this function
        if module == torch : 
            ct   = torch.from_numpy(self.contactTotal(t))
            ch   = torch.from_numpy(self.contactHome(t))
        else : 
            ct = self.contactTotal(t)
            ch = self.contactHome(t)

        b3 = 0.002 * self.lockdownLeakiness

        cl  = ct *  self.lockdownLeakiness     + ch * (1.0 - self.lockdownLeakiness)
        cl2 = ct * (self.lockdownLeakiness**2) + ch * (1.0 - self.lockdownLeakiness**2) 

        # lambda for non-lockdown
        current = ct * (i + a + self.beta2*e) / self.Nbar
        current += cl * (xi + xa + self.beta2*xe) / self.Nbar
        current[self.adultBins] += ct[self.adultBins, :] * b3 * p / self.Nbar[self.adultBins]
        lambdaNormal = module.sum(self.beta * current, axis=1)

        # lambda for lockdown
        current = cl * (i + a + self.beta2*e) / self.Nbar
        current += cl2 * (xi + xa + self.beta2*xe) / self.Nbar
        current[self.adultBins] += cl[self.adultBins, :] * b3 * p / self.Nbar[self.adultBins]
        lambdaLockdown = module.sum(self.beta * current, axis=1)

        # testing rates for presymptomatics, symptomatics and asymptomatics respectively
        # TODO : Ask Sanit what happened to tf2 and why are these values hardcoded
        testFrac1 = 3 * self.testingFraction1(t)/8
        testFrac2 = (5/3) * (testFrac1/(1-testFrac1))
        testFrac3 = self.testingFraction3(t)


        ds = -s * (lambdaNormal + self.k0(t)) + self.mu(t) * xs 
        de = self.f * lambdaNormal * s \
                - e * (self.k0(t) \
                    + self.gamma1 \
                    + testFrac3 * self.gamma1 /(1 - testFrac3)) \
                + self.mu(t) * xe 
        da = (1 - self.f) * lambdaNormal * s \
                - a * (self.k0(t) \
                    + self.sigma \
                    + testFrac1 * self.sigma/(1 - testFrac1)) \
                + self.mu(t) * xa 
        di = self.sigma * a \
                - i * (self.k0(t) \
                    + testFrac2 * self.gamma2 / (1 - testFrac2) \
                    + self.gamma2) \
                + self.mu(t) * xi 
        dxs = - xs * (lambdaLockdown + self.mu(t)) \
                + self.k0(t) * s
        dxe = self.f * lambdaLockdown * xs \
                + self.k0(t) * e \
                - xe * (self.mu(t) \
                    + self.gamma1 \
                    + testFrac3 * self.gamma1 /(1 - testFrac3))
        dxa = (1 - self.f) * lambdaLockdown * xs \
                - xa * (self.mu(t) \
                    + self.sigma \
                    + testFrac1 * self.sigma/(1 - testFrac1)) \
                + self.k0(t) * a 
        dxi = self.sigma * xa \
                + self.k0(t) * i \
                - xi * (self.mu(t) \
                    + testFrac2 * self.gamma2 / (1 - testFrac2) \
                    + self.gamma2)
        dp = testFrac2 * self.gamma2 / (1 - testFrac2) * (i + xi) \
                + testFrac1 * self.sigma/(1 - testFrac1) * (a + xa) \
                + testFrac3 * self.gamma1 /(1 - testFrac3) * (e + xe) \
                - self.gamma3 * p
        dr = self.gamma3 * p \
                + self.gamma2 * (i + xi) \
                + self.gamma1 * (e + xe) 

        
        self.setStates (s, e, a, i, xs, xe, xa, xi, p, r)
        return cat[module]((ds, de, da, di, dxs, dxe, dxa, dxi, dp, dr))

    def setBeta (self, b, ld) : 
        self.beta = b
        self.lockdownLeakiness = ld

    def setStates (self, s, e, a, i, xs, xe, xa, xi, p, r) : 
        self.s  = s
        self.e  = e
        self.a  = a
        self.i  = i
        self.xs = xs
        self.xe = xe
        self.xa = xa
        self.xi = xi
        self.p  = p
        self.r  = r

    def addCrossTerms (self, dx, module=np) : 
        ds, de, da, di, dxs, dxe, dxa, dxi, dp, dr = dx.reshape((-1, self.bins))

        ds[1]  += (self.sIn  - self.sOut)
        de[1]  += (self.eIn  - self.eOut)
        da[1]  += (self.aIn  - self.aOut)
        di[1]  += (self.iIn  - self.iOut)
        dr[1]  += (self.rIn  - self.rOut)

        return cat[module]((ds, de, da, di, dxs, dxe, dxa, dxi, dp, dr))

    def timeUpdate (self, x, t, module=np) : 
        dx = self.dx(x, t, module)
        return x + dx

if __name__ == "__main__" :
    with open('./Data/beta.json') as fd : 
        betas = json.load(fd)
    transportMatrix = np.loadtxt('./Data/transportMatrix.csv', delimiter=',')
    mortality = [getAgeMortality(s) for s in STATES]
    statePop  = [getStatePop(s) for s in STATES]
    model = IndiaModel(transportMatrix, betas, statePop, mortality) 
    x0 = []
    for Nbar in statePop : 
        N_ = deepcopy(Nbar)
        E0 = [0, 10, 0]
        A0 = [0, 10, 0]
        I0 = [0, 10, 0]
        ZE = [0, 0, 0]
        N_[1] -= 30
        x = [*N_, *E0, *A0, *I0, *ZE, *ZE, *ZE, *ZE, *ZE, *ZE]
        x0.extend(x)
    x0 = np.array(x0)
    results = linearApprox(model.dx, x0, 50)
    # results = results.T.reshape((len(STATES), 30, -1))
    # for r, s in zip(results, STATES) : 
    #     statePlot(r.T, s, Date('29 Feb'), 3)
