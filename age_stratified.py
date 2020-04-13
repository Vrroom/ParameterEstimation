
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv
import os

N, T, t_l, t_e, t_changeKt, t_changeKtDuration, kspread = None, None, None, None, None, None, None # Set later depending on country
t_changeContactStart, t_changeContactEnd = None, None
samplesPerDay = 1

state = 'MH'
startingTested = 14
daysOffset = 14

og_contactTotal = np.zeros((3, 3))
og_contactHome = np.zeros((3, 3))
## Read contact matrices
file = open('Data/total.csv')
lines = file.readlines()
file.close()
for i, line in enumerate(lines):
    for j, entry in enumerate(line.split(',')):
        og_contactTotal[i, j] = float(entry)
        
file = open('Data/home.csv')
lines = file.readlines()
file.close()
for i, line in enumerate(lines):
    for j, entry in enumerate(line.split(',')):
        og_contactHome[i, j] = float(entry)
        
if not os.path.exists('plots/' + state): os.mkdir('plots/' + state)


# In[ ]:


## THE SIER-X MODEL
def getPreds(beta = 0.2, lockdownLeakiness = 0.5, beta2 = 0.1, plot = False):
    global Nbar, T, t_l, t_e, t_changeKt, t_changeKtDuration, t_changeContactStart, t_changeContactEnd
    # Total population, N.

    # Initial number of infected and recovered individuals, I0 and R0.
    # Everyone else, S0, is susceptible to infection initially.

    sigma = 1/5.0
    K0 = 1./7
    MU = 1./7
    gamma1 = 1./21
    gamma2 = 1./21
    gamma3 = 1./19
    b3 = 0.002 * lockdownLeakiness
    f = 0.2
    numBins = 3
    adultBins = set([1])
    
#     A0, I0, Xs0, Xa0, Xi0, P0, R0 = 1, 0, 0, 0, 0, 0, 0
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

    t = np.linspace(0, T, samplesPerDay*T)
    
    def deriv(y, t, beta):
        assert len(y) == 10 * numBins
        S = list(y[0 * numBins : 1 * numBins])
        A = list(y[1 * numBins : 2 * numBins])
        I = list(y[2 * numBins : 3 * numBins])
        E = list(y[3 * numBins : 4 * numBins])
        Xs = list(y[4 * numBins : 5 * numBins])
        Xa = list(y[5 * numBins : 6 * numBins])
        Xi = list(y[6 * numBins : 7 * numBins])
        Xe = list(y[7 * numBins : 8 * numBins])
        P = list(y[8 * numBins : 9 * numBins])
        R = list(y[9 * numBins : 10 * numBins]) # can be represented as one, but kept for uniformity
        ############################################# LOCKDOWN ###########################################
        if t < t_l or t > t_e:
            k0 = 0
        else:
            k0 = K0
        if t > t_e:
            mu = MU
        else:
            mu = 0
        #######################################################################################################
        
        ############################################# RAMP UP TESTING ###########################################
        if t > t_changeKt + t_changeKtDuration:
            kt = 1.0; testingFraction1 = 0.8; testingFraction2 = 0.5; testingFraction3 = 0.5
        elif t > t_changeKt and t <= t_changeKt + t_changeKtDuration:
            weight = (t - t_changeKt) / t_changeKtDuration
            kt = 1.0 * weight + 0.5 * (1.0 - weight)
            testingFraction1 = 0.8 * weight + 1./13 * (1.0 - weight)
            testingFraction2 = testingFraction3 = 0.5 * weight + 1./13 * (1.0 - weight);
        else:
            kt = 0.5; testingFraction1 = 1./13; testingFraction2 = 0.0; testingFraction3 = 0.0
        #######################################################################################################
        
        ############################################# REDUCED CONTACT ###########################################
        if t < t_changeContactStart or t >= t_changeContactEnd:
            contactHome = og_contactHome
            contactTotal = og_contactTotal
        else:
            contactHome = 0.5 * og_contactHome
            contactTotal = 0.5 * og_contactTotal
        #######################################################################################################
        
        lamNormal, lamLockdown = [None] * 3, [None] * 3
        contactLockdown = contactTotal*lockdownLeakiness + contactHome*(1.0 - lockdownLeakiness)
        contactLockdown2 = contactTotal * (lockdownLeakiness**2) + contactHome * (1.0 - lockdownLeakiness**2)

        for i in range(numBins):
            # lambda_i for non-lockdown
            current = 0.0
            for j in range(numBins):
                current += contactTotal[i, j] * (I[j] + A[j] + beta2*E[j]) / Nbar[j]
            for j in range(numBins):
                current += contactLockdown[i, j] * (Xi[j] + Xa[j] + beta2*Xe[j]) / Nbar[j]
                if i in adultBins:
                    current += contactTotal[i, j] * b3 * P[j] / Nbar[j] #b3 should be there right?
            lamNormal[i] = beta * current
            # lambda_i for lockdown
            current = 0.0
            for j in range(numBins):
                current += contactLockdown[i, j] * (I[j] + A[j] + beta2*E[j]) / Nbar[j]
                current += contactLockdown2[i, j] * (Xi[j] + Xa[j] + beta2*Xe[j]) / Nbar[j]
                if i in adultBins:
                    current += contactLockdown[i, j] * b3 * P[j] / Nbar[j] #b3 should be there right?
            lamLockdown[i] = beta * current

        dSdt, dAdt, dIdt, dEdt = [], [], [], []
        dXsdt, dXadt, dXidt, dXedt, = [], [], [], []
        dPdt, dRdt = [], []
        for i in range(numBins):
            dSdt.append(- lamNormal[i] * S[i] - k0 * S[i] + mu * Xs[i])
            dAdt.append((1.0 - f) * lamNormal[i] * S[i] - (1.0 - testingFraction2) * sigma * A[i] - k0 * A[i] + mu * Xa[i] - testingFraction2 * kt * A[i])
            dIdt.append((1.0 - testingFraction2) * sigma * A[i]  - testingFraction1 * kt * I[i] - k0 * I[i] + mu * Xi[i] - gamma2 * (1 - testingFraction1) * I[i])
            dEdt.append(f * lamNormal[i] * S[i] - k0 * E[i] + mu * Xe[i] - (1.0 - testingFraction3) * gamma1 * E[i] - testingFraction3 * kt * E[i])

            dXsdt.append(- lamLockdown[i] * Xs[i] + k0 * S[i] - mu * Xs[i])
            dXadt.append((1.0 - f) * lamLockdown[i] * Xs[i] - (1.0 - testingFraction2) * sigma * Xa[i] + k0 * A[i] - mu * Xa[i] - testingFraction2 * kt * Xa[i])
            dXidt.append((1.0 - testingFraction2) * sigma * Xa[i] - testingFraction1 * kt * Xi[i] + k0 * I[i] - mu * Xi[i] - gamma2 * (1 - testingFraction1) * Xi[i])
            dXedt.append(f * lamLockdown[i] * Xs[i] + k0 * E[i] - mu * Xe[i] - (1.0 - testingFraction3) * gamma1 * Xe[i] - testingFraction3 * kt * Xe[i])

            dPdt.append(testingFraction1 * kt * (I[i] + Xi[i]) - gamma3 * P[i] + testingFraction2 * kt * (A[i] + Xa[i]) + testingFraction3 * kt * (E[i] + Xe[i]))
            dRdt.append(gamma2 * (1 - testingFraction1) * (I[i] + Xi[i]) + gamma3 * P[i] + (1.0 - testingFraction3) * gamma1 * (E[i] + Xe[i]))

        return tuple(dSdt + dAdt + dIdt + dEdt + dXsdt + dXadt + dXidt + dXedt + dPdt + dRdt)

    # Initial conditions vector
    y0 = Sbar0 + Abar0 + Ibar0 + Ebar0 + Xsbar0 + Xabar0 + Xibar0 + Xebar0 + Pbar0 + Rbar0
    y0 = tuple(y0)
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(beta,))
    S = ret.T[0 * numBins : 1 * numBins]
    A = ret.T[1 * numBins : 2 * numBins]
    I = ret.T[2 * numBins : 3 * numBins]
    E = ret.T[3 * numBins : 4 * numBins]
    Xs = ret.T[4 * numBins : 5 * numBins]
    Xa = ret.T[5 * numBins : 6 * numBins]
    Xi = ret.T[6 * numBins : 7 * numBins]
    Xe = ret.T[7 * numBins : 8 * numBins]
    P = ret.T[8 * numBins : 9 * numBins]
    R = ret.T[9 * numBins : 10 * numBins]
    P = P.sum(axis = 0)
    R = R.sum(axis = 0)
    I = I.sum(axis = 0)
    A = A.sum(axis = 0)
    Xi = Xi.sum(axis = 0)
    Xa = Xa.sum(axis = 0)
    if not plot: return P, R
    else: return P, I + Xi + P, P + I + Xi + A + Xa

# temp = getPreds(0.31, 0)


# In[ ]:


## READ DATA
def readDataIndia(startDate = None):
    # Start reading data from the startDate and subtract the number of confirmed, recovered and dead 
    # at the startDate from all future points
    file = open('data/time_series/india.csv', 'r')
    reader = csv.reader(file)
    confirmedOffset, recoveredOffset, deadOffset = 0, 0, 0
    confirmed, recovered, dead = None, None, None
    i, r = [], []
    for row in reader:
        if startDate is not None and (startDate.strip() == row[0].strip()):
            startDate = None
            confirmedOffset = int(row[2])
            recoveredOffset = int(row[4])
            deadOffset = int(row[6])
        if startDate is None:
            confirmed = int(row[2]) - confirmedOffset
            recovered = int(row[4]) - recoveredOffset
            dead = int(row[6]) - deadOffset
            r.append(recovered + dead)
            i.append(confirmed - r[-1])
    file.close()
    return i, r

def readDataState(startIndex = 0):
    global state
    file = open('Data/maha_data7apr.csv', 'r')
    reader = csv.reader(file)
    confirmed, recovered, dead = None, None, None
    count = -1
    i, r = [], []
    for row in reader:
        count += 1
        if count < startIndex: continue
        confirmed = int(row[2])
        recovered = int(row[4])
        dead = int(row[6])
        r.append(recovered + dead)
        i.append(confirmed - r[-1])
    file.close()
    return i, r

def getNbar(state):
    file = open('data/population/' + state + '.csv', 'r')
    lines = file.readlines()
    lines = [line.split(',') for line in lines if len(line) > 0]
    Nbar = [int(line[1]) for line in lines]
    assert len(Nbar) == 3
    file.close()
    return Nbar

# Maharashtra Data
# actualI, actualR = readDataState()
# actualI = np.asarray(actualI); actualR = np.asarray(actualR)
T = 50 - daysOffset
t_l = 24 - daysOffset
t_e = 24 + 21 - daysOffset
Nbar = [40544482., 60315220., 11106935.]
t_changeKt = 1000000000; t_changeKtDuration = 0
t_changeContactStart = 10000000000; t_changeContactEnd = 100000000000
kspread = 1.0


# In[ ]:


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
bestBeta, bestBeta1 = 0.015, 0.9
bestPreds = None

predI, predR = getPreds(beta = bestBeta, lockdownLeakiness = bestBeta1)
plt.plot(np.linspace(0, T, samplesPerDay*T), predI)
plt.show()

# In[ ]:


# def plot(pLock, totalLock, total2Lock, filename):
#     print("Total population of " + state + ": " + str(sum(Nbar)))
#     print("Maximum positive cases: " + str(int(pLock.max() + 0.5)))
#     print("Maximum symptomatic people: " + str(int(totalLock.max() + 0.5)))
#     print("Maximum infected people: " + str(int(total2Lock.max() + 0.5)))
# 
#     fig, (ax, ax2) = plt.subplots(1, 2, facecolor='w', figsize=(20, 7))
#     t = np.linspace(0, T, samplesPerDay * T + 1)
#     ax.plot(t[:len(pLock)], pLock, alpha=0.5, lw=2, label='Positive (P)')
#     ax.plot(t[:len(totalLock)], totalLock, lw=2, label='Positive + Symptomatic (P + $X_I$ + I)')
#     ax.plot(t[:len(total2Lock)], total2Lock, alpha=0.5, lw=2, label='Total Infected (P + $X_I$ + I + A + $X_A$)')
#     ax.set_xlabel('Time /days', fontsize=17)
#     ax.set_ylabel('Number of people', fontsize=17)
#     ax.set_yscale('log')
#     ax.set_ylim(1, 3e8)
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
#     ax.set_xticklabels(['', 'March 14', 'March 29', 'April 13', 'April 28', 'May 13', 'May 28', 'June 12', 'June 27',                       'July 12', 'July 27', 'August 11', 'August 26', 'Sept 10', 'Sept 25', 'Oct 10', 'Oct 25',                       'Nov 9', 'Nov 24', 'Dec 9', 'Dec 24', 'Jan 8'], rotation = 'vertical')
#     legend = ax.legend()
#     legend.get_frame().set_alpha(0.5)
#     ax.tick_params(labelsize=14)
#     ax.fill([10,31,31,10], [0,0,3 * 10**8,3 * 10**8], 'b', alpha=0.2, edgecolor='r')
#     ax.grid()
# 
#     t = np.linspace(0, T2, samplesPerDay * T2 + 1)
#     ax2.plot(t, pLock[:len(t)], alpha=0.5, lw=2, label='Positive (P)')
#     ax2.plot(t, totalLock[:len(t)], lw=2, label='Positive + Symptomatic (P + $X_I$ + I)')
#      #ax2.plot(t, total2Lock[:len(t)], alpha=0.5, lw=2, label='predicted P + Xi + i + A + Xa')
#     ax2.scatter(list(range(len(actualI))), actualI, c = 'r', label='Reported P')
#     ax2.set_xlabel('Time /days', fontsize=17)
#     ax2.set_ylabel('Number of people', fontsize=17)
#     ax2.xaxis.set_major_locator(ticker.MultipleLocator(3))
#     ax2.set_xticklabels(['', 'March 14', 'March 17', 'March 20', 'March 23', 'March 26', 'March 29', 'April 1',                         'April 4', 'April 7', 'April 10', 'April 13', 'April 16', 'April 19', 'April 22'], rotation = 'vertical')
#     legend = ax2.legend()
#     legend.get_frame().set_alpha(0.5)
#     ax2.tick_params(labelsize=14)
#     #ax2.fill([10,31,31,10], [0,0,3000,3000], 'b', alpha=0.2, edgecolor='r')
#     ax2.grid()
#     #fig.suptitle('Total population of Maharashtra: 114200000 \n Maximum positive cases:' + str(int(pLock.max() + 0.5))+'\n Maximum symptomatic people:'+ str(int(totalLock.max() + 0.5))+'\n Maximum infected people:' + str(int(total2Lock.max() + 0.5)), y=1.05,fontsize=16)
#     plt.savefig(filename, Transparent=True, bbox_inches='tight')
#     plt.show()
# 
# 
# # In[ ]:
# 
# 
# # beta = 0.31; beta1 = 0.25
# beta = bestBeta; beta1 = bestBeta1
# T = 200; T2 = 30
# kspread = 1.0
# 
# t_e = 24 + 21 - daysOffset
# t_changeKt = 45 - daysOffset
# t_changeKtDuration = 5
# pLock, totalLock, total2Lock = getPreds(beta, beta1, plot = True)
# plot(pLock, totalLock, total2Lock, 'plots/' + state + '/1a.png')
# 
# t_e = 24 + 21 - daysOffset
# t_changeKt = 1000000000
# pLock, totalLock, total2Lock = getPreds(beta, beta1, plot = True)
# plot(pLock, totalLock, total2Lock, 'plots/' + state + '/1b.png')
# 
# 
# # In[ ]:
# 
# 
# t_changeContactStart = 45 - daysOffset # from 14 April
# t_changeContactEnd = 31 + 30 + 16 - daysOffset # till 15 May
# 
# t_e = 24 + 21 - daysOffset
# t_changeKt = 45 - daysOffset
# t_changeKtDuration = 5
# pLock, totalLock, total2Lock = getPreds(beta, beta1, plot = True)
# plot(pLock, totalLock, total2Lock, 'plots/' + state + '/2a.png')
# 
# t_e = 24 + 21 - daysOffset
# t_changeKt = 1000000000
# pLock, totalLock, total2Lock = getPreds(beta, beta1, plot = True)
# plot(pLock, totalLock, total2Lock, 'plots/' + state + '/2b.png')
# 
# t_changeContactStart = 10000000000
# t_changeContactEnd = 1000000000
# 
# 
# # In[ ]:
# 
# 
# t_e = 31 + 30 + 1 - daysOffset # till 30 April
# t_changeKt = 45 - daysOffset
# t_changeKtDuration = 5
# pLock, totalLock, total2Lock = getPreds(beta, beta1, plot = True)
# plot(pLock, totalLock, total2Lock, 'plots/' + state + '/3a.png')
# 
# t_e = 31 + 30 + 1 - daysOffset # till 30 April
# t_changeKt = 1000000000
# pLock, totalLock, total2Lock = getPreds(beta, beta1, plot = True)
# plot(pLock, totalLock, total2Lock, 'plots/' + state + '/3b.png')
# 
# 
# # In[ ]:
# 
# 
# t_changeContactStart = 31 + 30 + 1 - daysOffset # from 1 May
# t_changeContactEnd = 31 + 30 + 16 - daysOffset # till 15 May
# 
# t_e = 31 + 30 + 1 - daysOffset # till 30 April
# t_changeKt = 45 - daysOffset
# t_changeKtDuration = 5
# pLock, totalLock, total2Lock = getPreds(beta, beta1, plot = True)
# plot(pLock, totalLock, total2Lock, 'plots/' + state + '/4a.png')
# 
# t_e = 31 + 30 + 1 - daysOffset # till 30 April
# t_changeKt = 1000000000
# pLock, totalLock, total2Lock = getPreds(beta, beta1, plot = True)
# plot(pLock, totalLock, total2Lock, 'plots/' + state + '/4b.png')
# 
# t_changeContactStart = 10000000000
# t_changeContactEnd = 1000000000
# 
# 
# # In[ ]:
# 
# 
# t_e = 31 + 30 + 15 + 1 - daysOffset # till 15 May
# t_changeKt = 45 - daysOffset
# t_changeKtDuration = 5
# pLock, totalLock, total2Lock = getPreds(beta, beta1, plot = True)
# plot(pLock, totalLock, total2Lock, 'plots/' + state + '/5a.png')
# 
# t_e = 31 + 30 + 15 + 1 - daysOffset # till 15 May
# t_changeKt = 1000000000
# pLock, totalLock, total2Lock = getPreds(beta, beta1, plot = True)
# plot(pLock, totalLock, total2Lock, 'plots/' + state + '/5b.png')
# 
