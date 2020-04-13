from Util import *
import math
from more_itertools import collapse
from functools import partial
import torch
import matplotlib.pyplot as plt
import numpy as np
from Simulate import *

class SpaxireAgeStratified () : 
    """
    Current ODE Model class. 
    
    The constructor takes a dictionary
    of parameters and initializes the model.
    """
    def __init__ (self, params) :
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

        self.names = list(more_itertools.collapse([
            [f'S{i}', f'E{i}', f'A{i}', f'I{i}', f'Xs{i}', f'Xe{i}', f'Xa{i}', f'Xi{i}', f'P{i}', f'R{i}'] 
            for i in range(self.bins)
        ]))
        self.colors = ['red', 'darkred', 'salmon',
                'chocolate', 'saddlebrown', 'sandybrown',
                'olive', 'lawngreen', 'green',
                'royalblue', 'blue', 'navy']
        self.cat = {np : np.hstack, torch : torch.cat}
        
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

        # Convert depending on usage of this function
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
                + self.testingFraction2(t) * self.kt(t) * (a + xi) \
                + self.testingFraction3(t) * self.kt(t) * (e + xe) \
                - self.gamma3 * p
        dr = self.gamma3 * p \
                + self.gamma2 * (1 - self.testingFraction1(t)) * (i + xi) \
                + (1 - self.testingFraction3(t)) * self.gamma1 * (e + xe)

        return self.cat[module]((ds, de, da, di, dxs, dxe, dxa, dxi, dp, dr))

    def timeUpdate (self, x, t, module=np) : 
        dx = self.dx(x, t, module)
        return x + dx

if __name__ == "__main__" :
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

    model = SpaxireAgeStratified(params)
    T = endDate - startDate
    x0 = [40544482.0, 60314940.0, 11106935.0, 0, 56.0, 0, 0, 210.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0]
    result = simulator(model, x0, np.linspace(0, T, T))
    plt.plot (np.linspace(0, T, T), result[:, -10])
    plt.show()
