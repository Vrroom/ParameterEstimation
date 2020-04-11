import torch
import numpy as np

class Spaxire () : 
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

        self.k0 = params['k0']
        self.kt = params['kt']
        self.mu = params['mu']

        self.beta  = params['beta']
        self.beta1 = params['beta1']
        self.beta2 = params['beta2']

        self.sigma  = params['sigma']
        self.gamma1 = params['gamma1']
        self.gamma2 = params['gamma2']
        self.gamma3 = params['gamma3']

        self.f = params['f']
        
        self.N = params['N']

        self.names = ['S', 'E', 'A', 'I', 'Xs', 'Xe', 'Xa', 'Xi', 'P', 'R']
        self.colors = ['red', 'blue', 'green', 'black', 
                'indianred', 'royalblue', 'lime', 
                'dimgray', 'orange', 'violet']
        
    def dx (self, x, t) : 
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
        """
        s, e, a, i, xs, xe, xa, xi, p, r = x

        k0 = 0 if t < self.tl or t > self.te else self.k0
        mu = self.mu if t > self.te else 0

        b1 = self.beta1
        b2 = 0.1   * self.beta1
        b3 = 0.002 * self.beta1

        # I have renamed lambda in the model document
        # to omega.
        omega = self.beta \
                * (i + a + b1 * xa + b2 * xi + b3 * p \
                + self.beta2 * (e + self.beta1 * xe))
        ds = -s * (omega / self.N + k0) + mu * xs
        de = self.f * omega * s / self.N - self.gamma1 * e
        da = (1 - self.f) * omega * s / self.N \
                - a * (self.sigma + k0) \
                + mu * xa
        di = self.sigma * a \
                - i * (self.kt + k0 + self.gamma2) \
                + mu * xi 
        dxs = - omega * self.beta1 * xs / self.N + k0 * s - mu * xs
        dxe = self.f * omega * self.beta1 * xs / self.N \
                - self.gamma1 * xe
        dxa = (1 - self.f) * omega * self.beta1 * xs/self.N \
                - xa * (self.sigma + mu) \
                + k0 * a 
        dxi = self.sigma * xa \
                - xi * (self.kt + mu + self.gamma2) \
                + k0 * i
        dp = self.kt * (i + xi) - self.gamma3 * p
        dr = self.gamma1 * (e + xe) \
                + self.gamma2 * (i + xi) \
                + self.gamma3 * p
        return ds, de, da, di, dxs, dxe, dxa, dxi, dp, dr

    def timeUpdate (self, x, t) : 
        dx = self.dx(x, t)
        if torch.is_tensor(x) : 
            return (torch.stack(dx) + x)
        else : 
            return (np.stack(dx) + x)
