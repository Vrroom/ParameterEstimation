from Util import *
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

        self.beta  = params['beta']
        self.beta1 = params['beta1']
        self.beta2 = params['beta2']

        self.sigma  = params['sigma']
        self.gamma1 = params['gamma1']
        self.gamma2 = params['gamma2']
        self.gamma3 = params['gamma3']

        self.testFraction = params['testFraction']
    
        self.bins = params['bins'] # Age bins
        self.adultBins = params['adultBins']
        
        self.f = params['f']
        self.N = params['N']
        self.Nbar = params['Nbar']

        self.k0  = params['k0']
        self.kt  = params['kt']
        self.kt2 = params['kt2']
        self.mu  = params['mu']

        self.lockdownLeakiness = params['lockdownLeakiness']

        self.contactHome = params['contactHome']
        self.contactTotal = params['contactTotal']

        self.contactLockdown = self.contactTotal*self.lockdownLeakiness + self.contactHome*(1.0 - self.lockdownLeakiness)
        self.contactLockdown2 = self.contactTotal * (self.lockdownLeakiness**2) + self.contactHome * (1.0 - self.lockdownLeakiness**2) 

        self.names = [
            [f'S{i}', f'E{i}', f'A{i}', f'I{i}', f'Xs{i}', f'Xe{i}', f'Xa{i}', f'Xi{i}', f'P{i}', f'R{i}'] 
            for i in range(self.bins)
        ]
        self.colors = ['red', 'blue', 'green', 'black', 
                'indianred', 'royalblue', 'lime', 
                'dimgray', 'orange', 'violet']

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
        s, e, a, i, xs, xe, xa, xi, p, r = x.reshape((self.bins, -1)).T

        # Convert depending on usage of this function
        if module == torch : 
            ct = self.contactTotal.numpy()
            cl = self.contactLockdown.numpy()
            cl2 = self.contactLockdown2.numpy()
            Nbar = self.Nbar.numpy()
        else : 
            ct = self.contactTotal
            cl = self.contactLockdown
            cl2 = self.contactLockdown2
            Nbar = self.Nbar

        b1 = self.beta1
        b2 = 0.1   * self.beta1
        b3 = 0.002 * self.beta1

        # lambda for non-lockdown
        current = ct * (i + a + self.beta2*e) / Nbar
        current += cl * (xi + xa + self.beta2*xe) / Nbar
        current[self.adultBins] += ct[self.adultBins, :] * b3 + p / Nbar[self.adultBins]
        lambdaNormal = np.sum(self.beta * current, axis=1)

        # lambda for lockdown
        current += cl * (i + a + self.beta2*e) / self.Nbar
        current += cl2 * (xi + xa + self.beta2*xe) / self.Nbar
        current[self.adultBins] += cl[self.adultBins, :] * b3 * p / self.Nbar[self.adultBins]
        lambdaLockdown = np.sum(self.beta * current, axis=1)

        ds = -s * (lambdaNormal / self.N + self.k0(t)) + self.mu(t) * xs
        de = self.f * lambdaNormal * s / self.N - self.gamma1 * e
        da = (1 - self.f) * lambdaNormal * s / self.N \
                - a * (self.sigma + self.k0(t)) \
                + self.mu(t) * xa
        di = self.sigma * a \
                - i * (self.testFraction * self.kt(t) + self.k0(t) + self.gamma2) \
                + self.mu(t) * xi 
        dxs = - xs * (lambdaLockdown * self.beta1 / self.N + self.mu(t)) \
                + self.k0(t) * s
        dxe = self.f * lambdaLockdown * self.beta1 * xs / self.N \
                + self.k0(t) * e \
                - xe * (self.gamma1 + self.mu(t))
        dxa = (1 - self.f) * lambdaLockdown * xs/self.N \
                - xa * (self.sigma + self.mu(t) + self.kt2(t)) \
                + self.k0(t) * a 
        dxi = self.sigma * xa \
                + self.k0(t) * i \
                - xi * (self.testFraction * self.kt(t) + self.mu(t) \
                        + (1 - self.testFraction) * self.gamma2)
        dp = self.testFraction * self.kt(t) * (i + xi) \
                - self.gamma3 * p \
                + self.kt2(t) * (a + xi)
        dr = self.gamma1 * (e + xe) \
                + self.gamma2 * (1 - self.testFraction) * (i + xi) \
                + self.gamma3 * p

        return self.cat[module]((ds, de, da, di, dxs, dxe, dxa, dxi, dp, dr))

    def timeUpdate (self, x, t, module=np) : 
        dx = self.dx(x, t, module)
        if torch.is_tensor(x) : 
            return (torch.stack(dx) + x)
        else : 
            return (np.stack(dx) + x)

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

    params = {
        'tl'                : lockdownBegin, 
        'te'                : lockdownEnd,
        'k0'                : lambda t : 0 if t < lockdownBegin or t > lockdownEnd else 1/7, 
        'kt'                : lambda t : 0.5 if t > changeKt else 0.075,
        'kt2'               : lambda t : 0.5 if t > changeKt else 0.075,
        'mu'                : lambda t : 1/7 if t > lockdownEnd else 0,
        'sigma'             : 1/5,
        'gamma1'            : 1/21,
        'gamma2'            : 1/21,
        'gamma3'            : 1/17,
        'N'                 : 1.1e8,
        'beta'              : 0.16,
        'beta1'             : 1.8,
        'beta2'             : 0.1,
        'f'                 : 0.1,
        'lockdownLeakiness' : 0.5,
        'contactHome'       : contactHome,
        'contactTotal'      : contactTotal,
        'testFraction'      : 1/13,
        'bins'              : 3,
        'Nbar'              : np.array([40544482., 60315220., 11106935.]),
        'adultBins'         : [1]
    }

    model = SpaxireAgeStratified(params)
    T = endDate - startDate
    result = simulator(model, np.zeros(30), np.arange(0, T, 0.5))
    plt.plot (np.arange(0, T, 0.5), result[:, 0])
    plt.show()
