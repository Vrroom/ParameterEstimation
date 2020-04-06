from Search import *
from Losses import *
from itertools import product
import numpy as np
import pandas

if __name__ == "__main__" : 
    paramRanges = {
        'tl'    : [10], 
        'te'    : [31],
        'k0'    : [1/7], 
        'kt'    : [0.075],
        'mu'    : [1/7],
        'sigma' : np.arange(0, 0.5, 0.05),
        'gamma1': [1/19],
        'gamma2': [1/14],
        'gamma3': [1/14],
        'N'     : [1.1e8],
        'beta'  : np.arange(0, 1, 0.05),
        'beta1' : np.arange(0, 1, 0.05)
    }
    I0  = np.arange(1, 5000, 100)
    A0  = np.arange(0, 5000, 100)
    S0  = list(map(lambda x : 1.1e8 - x[0] - x[1], product(I0, A0)))
    Xs0 = [0]
    Xa0 = [0]
    Xi0 = [0]
    P0  = [0]
    R0  = [0]
    ivRanges = [S0, A0, I0, Xs0, Xa0, Xi0, P0, R0]
    data = pandas.read_csv('./Data/maha_data.csv')
    lastDateIdx = data[data['Date'] == '31 Mar'].index[0]
    data = data.iloc[:lastDateIdx]
    result = gridSearch(ivRanges, paramRanges, data, squaredLoss, np.arange(3, 31) - 3)
    print(result)
