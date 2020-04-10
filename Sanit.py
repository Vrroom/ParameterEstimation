import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import csv

samplesPerDay = 10
def getPreds(beta = 0.2, beta1 = 0.1):
    # Total population, N.
    N = 1.1e8
    # lockdown imposition
    K0 = 1/7
    # lifting the lockdown
    MU = 1/7  
    beta2 = 0.1
    # fraction asymptomatic
    f = 0.1
    # recovery rates
    gamma1 = 1/21
    gamma2 = 1/21
    gamma3 = 1/17
    # incubation
    sigma = 1/5
    # testing rate
    kt = 0.075#3113.0/79950
    # intervention effectiveness coefficients
    b1 = beta1
    b2 = 0.1 * beta1
    b3 = 0.002 * beta1
    # initial conditions
    P0 = 14
    init_ratio = 20
    totalInfected = P0 * init_ratio
    E0 = totalInfected * f
    A0 = totalInfected * (1 - f) - P0
    I0 = 0
    Xs0 = 0
    Xe0 = 0
    Xa0 = 0
    Xi0 = 0
    R0 = 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - E0 - A0 - I0 - Xs0 - Xe0 - Xa0 - Xi0 - P0 - R0
    t = np.linspace(0, T, samplesPerDay*T)
    def deriv(y, t, N, beta):
        S, E, A, I, Xs, Xe, Xa, Xi, P, R = y
        if t < t_l or t > t_e:
            k0 = 0
        else:
            k0 = K0
        if t > t_e:
            mu = MU
        else:
            mu = 0
        lamb = beta * (I + A + b1 * Xa + b2 * Xi + b3 * P + beta2 * (E + beta1 * Xe))
        dSdt = - lamb * S/N - k0 * S + mu * Xs
        dEdt = f * lamb * S/N - gamma1 * E
        dAdt = (1 - f) * lamb * S/N - sigma * A - k0 * A + mu * Xa
        dIdt = sigma * A  - kt * I - k0 * I + mu * Xi - gamma2 * I
        dXsdt = - lamb * beta1 * Xs/N + k0 * S - mu * Xs
        dXedt = f * lamb * beta1 * Xs/N - gamma1 * Xe
        dXadt = (1-f) * lamb * beta1 * Xs/N - sigma * Xa + k0 * A - mu * Xa
        dXidt = sigma * Xa - kt * Xi + k0 * I - mu * Xi - gamma2 * Xi
        dPdt = kt * (I + Xi) - gamma3 * P
        dRdt =  + gamma2 * (I + Xi) + gamma3 * P + gamma1 * (E + Xe)
        print(dSdt, dEdt, dAdt, dIdt, dXsdt, dXedt, dXadt, dXidt, dPdt, dRdt)
        return dSdt, dEdt, dAdt, dIdt, dXsdt, dXedt, dXadt, dXidt, dPdt, dRdt
    # Initial conditions vector
    y0 = S0, E0, A0, I0, Xs0, Xe0, Xa0, Xi0, P0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta))
    S, E, A, I, Xs, Xe, Xa, Xi, P, R = ret.T
    return P, R

## READ DATA
def readDataIndia(startDate = None):
    # Start reading data from the startDate and subtract the number of confirmed, recovered and dead 
    # at the startDate from all future points
    file = open('./Data/maha_data7apr.csv', 'r')
    reader = csv.reader(file)
    confirmedOffset, recoveredOffset, deadOffset = 0, 0, 0
    confirmed, recovered, dead = None, None, None
    i, r = [], []
    for row in reader:
#         if startDate is not None and (startDate.strip() == row[0].strip()):
#             startDate = None
#             confirmedOffset = int(row[2])
#             recoveredOffset = int(row[4])
#             deadOffset = int(row[6])
        if startDate is None:
            confirmed = int(row[2]) - confirmedOffset
            recovered = int(row[4]) - recoveredOffset
            dead = int(row[6]) - deadOffset
            r.append(recovered + dead)
            i.append(confirmed - r[-1])
    file.close()
    return i, r

actualI, actualR = readDataIndia()
T = 36
t_l = 40# From 24 March, ignore Janta Curfew?
t_e = 42
actualI = np.asarray(actualI); actualR = np.asarray(actualR)
bestI, bestR = getPreds(beta = 0.16, beta1 = 1.8)
## Plot
fig = plt.figure(facecolor='w', figsize=(15, 10))
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
t = np.linspace(0, T, samplesPerDay * T + 1)
# ax.plot(t[:len(bestR)], bestR, 'g', alpha=0.5, lw=2, label='predicted R')
ax.plot(t[:len(bestI)], bestI, 'r', alpha=0.5, lw=2, label='predicted P')
ax.scatter(list(range(len(actualI))), actualI, c = 'r', label='actual P')
# ax.scatter(list(range(len(actualR))), actualR, c = 'g', label='actual R')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number(\% pop)')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
plt.show()


