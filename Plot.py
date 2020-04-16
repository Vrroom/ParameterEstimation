import matplotlib.pyplot as plt
from matplotlib import ticker
from Util import *
import pandas
from Model import *
from Simulate import *
from more_itertools import collapse

def gather(T, series, variances, indices):
    output = np.zeros(())

def statePlot (series, variances, state, beginDate, step) : 
    print(variances[0].shape)
    T, _ = series.shape
    compartments = ['S', 'E', 'A', 'I', 'Xs', 'Xe', 'Xa', 'Xi', 'P', 'R']
    bins = ['0-20', '20-60', '60+']
    series = series.T.reshape((10, -1, T))
    std = np.sqrt(variances.T.reshape((10, -1, T)))
    fig, ax = plt.subplots(nrows=5, ncols=2, sharex=True, figsize=(20, 15))
    fig.suptitle(state)
    tickLabels = list(DateIter(beginDate, beginDate + T))[::step]
    tickLabels = [d.date for d in tickLabels]
    days = np.array([np.arange(T) for _ in range(3)])
    for a, var, st, c in zip(collapse(ax), series, std, compartments): 
        labels = [c + '_' + b for b in bins]
        lines = a.plot(days.T, var.T)
        a.legend(lines, labels)
        for v, s, l in zip(var, st, lines) : 
            a.fill_between(np.arange(T), np.maximum(v-s, 0), v+s, facecolor=l.get_c(), alpha=0.2)
        a.set_xlabel('Time / days')
        a.set_ylabel('Number of people')
        a.set_yscale('log')
        a.xaxis.set_major_locator(ticker.MultipleLocator(step))
        a.set_xticklabels(tickLabels, rotation = 'vertical')
    fig.savefig('./Plots/' + state)
    plt.close(fig)
