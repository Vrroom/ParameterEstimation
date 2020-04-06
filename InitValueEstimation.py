import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from Util import *

T, t_l, t_e = None, None, None # Set later depending on country
samplesPerDay = 10

## THE SIER-X MODEL
def getPreds(beta = 0.2, beta1 = 0.1):
    global T, t_l, t_e
    # Total population, N.
    N = 110000000

    # Initial number of infected and recovered individuals, I0 and R0.
#     A0, I0, Xs0, Xa0, Xi0, P0, R0 = 1, 0, 0, 0, 0, 0, 0
    totalInfected = 14 * 20; pIgivenA = 0.33
    A0 = totalInfected * (1.0 - pIgivenA) ;I0 = totalInfected * pIgivenA ;Xs0 = 0;Xa0 = 0;Xi0 = 0;P0 = 14; R0 = 0

    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - A0 - I0 - Xs0 - Xa0 - Xi0 - P0 - R0

    K0 = 1./7
    MU = 1./7
    gamma1 = 1./19
    sigma = gamma1/2.0
    gamma2 = 1./14
    gamma3 = 1./14
    kt = 0.075#3113.0/79950
    b1 = beta1
    b2 = 0.1 * beta1
    b3 = 0.002 * beta1

    t = np.linspace(0, T, samplesPerDay*T)

    def deriv(y, t, N, beta):
        S, A, I, Xs, Xa, Xi, P, R = y


        if t < t_l or t > t_e:
            k0 = 0
        else:
            k0 = K0

        if t > t_e:
            mu = MU
        else:
            mu = 0


        dSdt = - beta * (I + A + b1 * Xa + b2 * Xi + b3 * P) * S/N - k0 * S + mu * Xs
        dAdt = beta * (I + A + b1 * Xa + b2 * Xi + b3 * P) * S/N - sigma * A - k0 * A + mu * Xa - gamma1 * A
        dIdt = sigma * A  - kt * I - k0 * I + mu * Xi - gamma2 * I

        dXsdt = - beta * beta1 * (I + A + b1 * Xa + b2 * Xi + b3 * P) * Xs/N + k0 * S - mu * Xs
        dXadt = beta * beta1 * (I + A + b1 * Xa + b2 * Xi + b3 * P) * Xs/N - sigma * Xa + k0 * A - mu * Xa - gamma1 * Xa
        dXidt = sigma * Xa - kt * Xi + k0 * I - mu * Xi - gamma2 * Xi

        dPdt = kt * (I + Xi) - gamma3 * P
        dRdt = gamma1 * (A + Xa) + gamma2 * (I + Xi) + gamma3 * P

        return dSdt, dAdt, dIdt, dXsdt, dXadt, dXidt, dPdt, dRdt

    # Initial conditions vector
    y0 = S0, A0, I0, Xs0, Xa0, Xi0, P0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta))
    S, A, I, Xs, Xa, Xi, P, R = ret.T
#     print(Xs)
    return P, R

# temp = getPreds(0.31, 0)

## READ DATA
# # Start from March 01, when the first case in this spread started, adjust counts as needed
# actualI, actualR = readDataIndia(startDate = "01 March")
# T = 50
# t_l = 23 # From 24 March, ignore Janta Curfew?
# t_e = 44
# actualI = np.asarray(actualI[15:]); actualR = np.asarray(actualR[15:])

data = getInfectedAndRecovered('maha_data2.csv')
actualI, actualR = data[0], data[1]

T = 30
t_l = 10# From 24 March, ignore Janta Curfew?
t_e = 31

# Define loss and search for the best hyperparameters (grid search for now)
def squaredLoss(preds, target):
    loss = ((preds - target) ** 2).sum()
    return loss

eps = 1e-8
def squaredLossExpScale(preds, target):
    preds = np.log(preds + eps)
    target = np.log(target + eps)
    loss = ((preds - target) ** 2).sum()
    return loss

bestLoss = 10000000000
bestBeta, bestBeta1 = -1, -1
bestPreds = None
# betaValues = [0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.36]
betaValues = np.linspace(0.2, 0.5, 31)
beta1Values = np.linspace(0.5, 1.0, 21)
losses = np.zeros((len(betaValues), len(beta1Values)))

for i, beta in enumerate(betaValues):
    for j, beta1 in enumerate(beta1Values):
        predI, predR = getPreds(beta = beta, beta1 = beta1)
        import pdb
        pdb.set_trace()
        trimmedPredI = predI[:len(actualI) * samplesPerDay:samplesPerDay]
#         trimmedPredR = predR[:len(actualR) * samplesPerDay:samplesPerDay]
        assert len(trimmedPredI) == len(actualI), "Length mismatch"
#             loss = squaredLossExpScale(trimmedPredI, actualI) + squaredLossExpScale(trimmedPredR, actualR)
        loss = squaredLoss(trimmedPredI, actualI)# + squaredLoss(trimmedPredR, actualR)
        losses[i, j] = loss

        if loss < bestLoss:
            bestLoss = loss
            bestPreds = predI, predR
            bestBeta = beta; bestBeta1 = beta1

print("Best beta value", bestBeta)
print("Best beta1 value", bestBeta1)

## Plot
bestI, bestR = bestPreds
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
