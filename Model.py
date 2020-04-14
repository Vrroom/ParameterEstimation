from Util import *
import json
import random
import math
from more_itertools import collapse
from itertools import product
from functools import partial
import torch
import matplotlib.pyplot as plt
import numpy as np
from Simulate import *
from copy import deepcopy

# TODO : LOAD POP for each state BOTH IN MODULE AND IN
# INIT
# TODO : FIX INITIAL CONDITIONS
cat = {np : np.hstack, torch : torch.cat}

class IndiaModel () : 

    STATES = ['ANDAMAN&NICOBARISLANDS', 'ANDHRAPRADESH', 'ARUNACHALPRADESH', 
            'ASSAM', 'BIHAR', 'CHANDIGARH', 'CHHATTISGARH', 
            'DADRA&NAGARHAVELI', 'DAMAN&DIU', 'GOA', 
            'GUJARAT', 'HARYANA', 'HIMACHALPRADESH', 
            'JAMMU&KASHMIR', 'JHARKHAND', 'KARNATAKA', 
            'KERALA', 'LAKSHADWEEP', 'MADHYAPRADESH',
            'MAHARASHTRA', 'MANIPUR', 'MEGHALAYA', 
            'MIZORAM', 'NAGALAND', 'NCTOFDELHI', 
            'ODISHA', 'PUDUCHERRY', 'PUNJAB', 
            'RAJASTHAN', 'SIKKIM', 'TAMILNADU', 
            'TRIPURA', 'UTTARAKHAND', 'UTTARPRADESH', 'WESTBENGAL']

    def __init__ (self, transportMatrix, betas) : 
        self.transportMatrix = transportMatrix
        self.betas = betas
        self.bins = 3
        self.states = len(self.STATES)
        self.setStateModels()

    def dx (self, x, t, module=np) : 
        xs = x.reshape((self.states, -1))
        derivatives = [m.dx(x, t, module) for x, m in zip(xs, self.models)]
        
        for m in self.models : 
            m.send()

        for i in range(self.states) : 
            _, outChannel = self.links[i]
            data = outChannel.pop()
            for j in range(self.states) :
                data_ = deepcopy(data)
                for key in data.keys() : 
                    data_[key] *= self.transportMatrix[i, j]
                inChannel, _ = self.links[j]
                inChannel.append(data_)

        for m in self.models : 
            m.receive()

        derivatives = [m.addCrossTerms(dx, module) for dx in derivatives]
        return cat[module](derivatives)

    def setStateModels (self):
        startDate = Date('29 Feb')
        firstCases = Date('14 Mar')
        firstDeath = Date('17 Mar')
        endDate = Date('7 Apr')

        changeKt = Date('27 Mar') - startDate

        lockdownBegin = Date('24 Mar') - startDate
        lockdownEnd = Date('14 Apr') - startDate

        contactHome = np.loadtxt('./Data/home.csv', delimiter=',')
        contactTotal = np.loadtxt('./Data/total.csv', delimiter=',')

        changeContactStart = math.inf
        changeContactEnd   = math.inf

        changeKt = math.inf
        deltaKt  = math.inf

        params = {
            'tl'                : lockdownBegin, 
            'te'                : lockdownEnd,
            'k0'                : partial(bumpFn, ti=lockdownBegin, tf=lockdownEnd, x1=0, x2=1/7),
            'kt'                : partial(climbFn, ti=changeKt, tf=changeKt+deltaKt, xi=0.5, xf=1.0),
            'mu'                : partial(stepFn, t0=lockdownEnd, x1=0, x2=1/7),
            'sigma'             : 1/5,
            'gamma1'            : 1/21,
            'gamma2'            : 1/21,
            'gamma3'            : 1/19,
            'N'                 : 1.1e8,
            'beta'              : 0.015,
            'beta2'             : 0.1,
            'f'                 : 0.2,
            'lockdownLeakiness' : 0.9,
            'contactHome'       : partial(bumpFn, ti=changeContactStart, tf=changeContactEnd, x1=contactHome, x2=0.5*contactHome),
            'contactTotal'      : partial(bumpFn, ti=changeContactStart, tf=changeContactEnd, x1=contactTotal, x2=0.5*contactTotal),
            'bins'              : 3,
            'Nbar'              : np.array([40544482., 60315220., 11106935.]),
            'adultBins'         : [1],
            'testingFraction1'  : partial(climbFn, ti=changeKt, tf=changeKt+deltaKt, xi=1/13, xf=0.8),
            'testingFraction2'  : partial(climbFn, ti=changeKt, tf=changeKt+deltaKt, xi=0, xf=0.5),
            'testingFraction3'  : partial(climbFn, ti=changeKt, tf=changeKt+deltaKt, xi=0, xf=0.5),
        }

        self.models = []
        self.links = []
        for idx, state in enumerate(self.STATES) : 
            beta, lockdownLeakiness = self.betas[state]
            p = deepcopy(params)
            p['beta'] = beta
            p['lockdownLeakiness'] = lockdownLeakiness
            p['totalOut'] = self.transportMatrix[idx].sum()
            inChannel, outChannel = [], []
            self.links.append((inChannel, outChannel))
            self.models.append(SpaxireAgeStratified(p, self.transportMatrix, inChannel, outChannel))

class SpaxireAgeStratified () : 
    """
    Current ODE Model class. 
    
    The constructor takes a dictionary
    of parameters and initializes the model.
    """
    def __init__ (self, params, idx, inChannel, outChannel) :
        """
        ODE has a lot of parameters.
        These are present in a dictionary from
        which the model is initialized.

        Parameters
        ----------
        params : dictionary of parameters
            Many of the parameters are easy to
            fix because they are determined by
            the COVID situation in India. For 
            example kt is the testing rate. 
            Other parameters such as beta/beta1
            which are related to how the disease
            spreads aren't so easy to specify.
        """
        self.idx = idx # Index of this state

        self.inChannel = inChannel
        self.outChannel = outChannel

        self.tl = params['tl']
        self.te = params['te']

        self.k0  = params['k0']
        self.kt  = params['kt']
        self.mu  = params['mu']

        self.sigma  = params['sigma']
        self.gamma1 = params['gamma1']
        self.gamma2 = params['gamma2']
        self.gamma3 = params['gamma3']

        self.N = params['N']

        self.beta  = params['beta']
        self.beta2 = params['beta2']

        self.f = params['f']
        self.lockdownLeakiness = params['lockdownLeakiness']

        self.contactHome = params['contactHome']
        self.contactTotal = params['contactTotal']
    
        self.bins = params['bins'] # Age bins
        self.Nbar = params['Nbar']
        self.adultBins = params['adultBins']

        self.testingFraction1 = params['testingFraction1']
        self.testingFraction2 = params['testingFraction2']
        self.testingFraction3 = params['testingFraction3']

        self.totalOut = params['totalOut']

        names = ['S', 'E', 'A', 'I', 'Xs', 'Xe', 'Xa', 'Xi', 'P', 'R']
        self.names = [[n + str(i) for i in range(1, self.bins + 1)] for n in names]
        self.names = list(collapse(self.names))
        
        r = [random.random() for _ in range(30)]
        g = [random.random() for _ in range(30)]
        b = [random.random() for _ in range(30)]

        self.colors = list(zip(r,g,b))

    def send (self) : 
        bottom = self.s + self.e + self.a + self.i + self.r

        sOut = self.s / bottom
        eOut = self.e / bottom
        aOut = self.a / bottom
        iOut = self.i / bottom
        rOut = self.r / bottom 

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

    def dx (self, x, t, module) : 
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
            Nbar = torch.from_numpy(self.Nbar)
        else : 
            ct = self.contactTotal(t)
            ch = self.contactHome(t)
            Nbar = self.Nbar

        b3 = 0.002 * self.lockdownLeakiness

        cl  = ct *  self.lockdownLeakiness     + ch * (1.0 - self.lockdownLeakiness)
        cl2 = ct * (self.lockdownLeakiness**2) + ch * (1.0 - self.lockdownLeakiness**2) 

        # lambda for non-lockdown
        current = ct * (i + a + self.beta2*e) / Nbar
        current += cl * (xi + xa + self.beta2*xe) / Nbar
        current[self.adultBins] += ct[self.adultBins, :] * b3 * p / Nbar[self.adultBins]
        lambdaNormal = module.sum(self.beta * current, axis=1)

        # lambda for lockdown
        current = cl * (i + a + self.beta2*e) / Nbar
        current += cl2 * (xi + xa + self.beta2*xe) / Nbar
        current[self.adultBins] += cl[self.adultBins, :] * b3 * p / Nbar[self.adultBins]
        lambdaLockdown = module.sum(self.beta * current, axis=1)

        ds = -s * (lambdaNormal + self.k0(t)) + self.mu(t) * xs 
        de = self.f * lambdaNormal * s \
                - e * (self.k0(t) \
                    + (1 - self.testingFraction3(t)) * self.gamma1 \
                    + self.testingFraction3(t) * self.kt(t)) \
                + self.mu(t) * xe 
        da = (1 - self.f) * lambdaNormal * s \
                - a * (self.k0(t) \
                    + (1 - self.testingFraction2(t)) * self.sigma \
                    + self.testingFraction2(t) * self.kt(t)) \
                + self.mu(t) * xa 
        di = (1 - self.testingFraction2(t)) * self.sigma * a \
                - i * (self.k0(t) \
                    + self.testingFraction1(t) * self.kt(t) \
                    + (1 - self.testingFraction1(t)) * self.gamma2) \
                + self.mu(t) * xi 
        dxs = - xs * (lambdaLockdown + self.mu(t)) \
                + self.k0(t) * s
        dxe = self.f * lambdaLockdown * xs \
                + self.k0(t) * e \
                - xe * (self.mu(t) \
                    + (1 - self.testingFraction3(t)) * self.gamma1 \
                    + self.testingFraction3(t) * self.kt(t))
        dxa = (1 - self.f) * lambdaLockdown * xs \
                - xa * (self.mu(t) \
                    + (1 - self.testingFraction2(t)) * self.sigma \
                    + self.testingFraction2(t) * self.kt(t)) \
                + self.k0(t) * a 
        dxi = (1 - self.testingFraction2(t)) * self.sigma * xa \
                + self.k0(t) * i \
                - xi * (self.mu(t) \
                    + self.testingFraction1(t) * self.kt(t) \
                    + (1 - self.testingFraction1(t)) * self.gamma2)
        dp = self.testingFraction1(t) * self.kt(t) * (i + xi) \
                + self.testingFraction2(t) * self.kt(t) * (a + xa) \
                + self.testingFraction3(t) * self.kt(t) * (e + xe) \
                - self.gamma3 * p
        dr = self.gamma3 * p \
                + self.gamma2 * (1 - self.testingFraction1(t)) * (i + xi) \
                + (1 - self.testingFraction3(t)) * self.gamma1 * (e + xe) 

        self.setStates (s, e, a, i, xs, xe, xa, xi, p, r)
        return cat[module]((ds, de, da, di, dxs, dxe, dxa, dxi, dp, dr))

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

        ds  += self.s  * (self.sIn  - self.sOut )
        de  += self.e  * (self.eIn  - self.eOut )
        da  += self.a  * (self.aIn  - self.aOut )
        di  += self.i  * (self.iIn  - self.iOut )
        dr  += self.r  * (self.rIn  - self.rOut )

        return cat[module]((ds, de, da, di, dxs, dxe, dxa, dxi, dp, dr))

    def timeUpdate (self, x, t, module=np) : 
        dx = self.dx(x, t, module)
        return x + dx

def linearApprox (fn, x0, T) : 
    out = [x0]
    x = x0
    for t in range(T) : 
        x = x + fn(x, t)
        out.append(x)
    return np.array(out)

if __name__ == "__main__" :
    with open('./Data/beta.json') as fd : 
        betas = json.load(fd)
    transportMatrix = np.loadtxt('./Data/transportMatrix.csv', delimiter=',')
    model = IndiaModel(transportMatrix, betas) 
    x0 = np.ones(1050)
    results = linearApprox(model.dx, x0, 4)
    plt.plot(range(5), results[:, 0])
    plt.show()
