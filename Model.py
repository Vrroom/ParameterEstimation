
class Sixer () : 

    def __init__ (self, init, params) :
        self.init = init

        self.tl = params['tl']
        self.te = params['te']

        self.k0 = params['k0']
        self.mu = params['mu']

        self.beta = params['beta']
        self.beta1 = params['beta1']

        self.sigma = params['sigma']
        self.gamma1 = params['gamma1']
        self.gamma2 = params['gamma2']
        self.gamma3 = params['gamma3']
        
        self.N = params['N']
        
    def dx (self, y, t) : 
        s, a, i, xs, xa, xi, p, r = y

        k0 = 0 if t < self.tl or t > self.te else self.k0
        mu = self.mu if t > self.te else 0

        b1 = self.beta1
        b2 = 0.1   * self.beta1
        b3 = 0.002 * self.beta1

        factor = (i + a + b1 * xa + b2 * xi + b3 * p)

        ds = -self.beta * factor * s/self.n \
                - k0 * s \
                + mu * xs
        da = self.beta * factor * s/self.N \
                - a * (self.sigma + k0 + self.gamma1) \
                + mu * xa
        di = self.sigma * a \
                - i * (self.kt + k0 + self.gamma2) \
                + mu * xi 
        dxs = -self.beta * self.beta1 * factor * xs/self.N \
                + k0 * s - mu * xs
        dxa = self.beta * self.beta1 * factor * xs/self.N \
                - xa * (self.sigma + self.gamma1 + mu) \
                + k0 * a 
        dxi = self.sigma * xa \
                - xi * (kt + mu + self.gamma2) \
                + k0 * i
        dp = kt * (i + xi) - self.gamma3 * p
        dr = self.gamma1 * (a + xa) \
                + self.gamma2 * (i + xi) \
                + self.gamma3 * p
        return ds, da, di, dxs, dxa, dxi, dp, dr

