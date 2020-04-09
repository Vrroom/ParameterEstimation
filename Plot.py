import matplotlib.pyplot as plt
from matplotlib import ticker
from Util import *
import pandas
from Model import *
from Simulate import *

N = 1.1e8
params = {
    'tl': 21, 
    'te': 42, 
    'k0': 0.14285714285714285, 
    'kt': 0.075, 
    'mu': 0.14285714285714285, 
    'sigma': 0.30000000000000004, 
    'gamma1': 0.05263157894736842, 
    'gamma2': 0.07142857142857142, 
    'gamma3': 0.07142857142857142, 
    'N': 110000000.0, 
    'beta': 0.3500000000000001, 
    'beta1': 0.1
}
iv = [10, 1, 0, 0, 0, 0, 0]
iv = [N - sum(iv), *iv]
model = Sixer(iv, params)
samplesPerDay = 10
T = 30
result = simulator(model, np.linspace(0, T, T * samplesPerDay))

ir = getInfectedAndRecovered('Data/maha_data.csv')
pLock = result[:, -2]
totalLock = result[:, -2] + result[:, 2] + result[:, -3]
total2Lock = result[:, -2] + result[:, 2] + result[:, -3] + result[:, 1]

fig, ax = plt.subplots()
t = np.linspace(0, T, samplesPerDay * T)
ax.plot(t, pLock[:len(t)], alpha=0.5, lw=2, label='predicted P')
ax.plot(t, totalLock[:len(t)], lw=2, label='predicted P + Xi + i')
ax.scatter(range(11, 11 + len(ir[:,0])), ir[:,0], c = 'r', label='actual P')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number of people')
# ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
# labels = [
#     '', 'March 3', 'March 4', 'March 5', 'March 6', 'March 7', 'March 8', 'March 9', \
#     'March 10', 'March 11', 'March 12', 'March 13','March 14', 'March 17', 'March 20',\
#     'March 23', 'March 27', 'March 30', 'March 31', \
# ]
# ax.set_xticklabels(labels, rotation = 'vertical')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
plt.show()
