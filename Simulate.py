from Util import *
import Plot
from EKF import *
import json
import Model
from scipy.integrate import odeint
from functools import partial
import numpy as np 
import pandas

def odeSimulator (model, x0, T) :
    dx = partial(model.dx, module=np)
    result = odeint(dx, x0, T)
    return result

class KalmanSimulator () :

    def __init__ (self, data, model, x0) : 
        self.x0 = x0

        self.data = data
        self.model = model
        self.dates = data['Date'].map(self.splitDates)

        self.firstCases = Date(self.dates.iloc[0])
        self.dataEndDate = Date(self.dates.iloc[-1])

        self.peopleDied = self.dates[data['Total Dead'] > 0].size > 0
        if self.peopleDied : 
            self.firstDeath = Date(self.dates[data['Total Dead'] > 0].iloc[0])
            self.startDate = self.firstDeath - 17
            self.deaths = self.data['Daily Dead'][data['Total Dead'] > 0].to_numpy()
        else : 
            self.startDate = self.firstCases

        self.P = (data['Total Cases'] - data['Total Recovered'] - data['Total Dead']).to_numpy()
        
        self.h1, self.h2 = [0] * 30, [0] * 30 
        self.h1[9:12] = model.mortality.tolist() # Setting mortality

        self.h1[21:24] = model.mortality.tolist() # Setting mortality

        self.h1[24:27] = model.mortality.tolist() # Setting mortality
        self.h2[-6:-3] = [1,1,1] # Setting P

        self.setP0() 
        self.setQ()

    def setP0(self) : 
        self.P0 = np.diag([1e2] * 12 + [1] * 18)

    def setQ (self) :
        self.Q = np.diag([1e1] * 24 + [1] * 6)
    
    def splitDates (self, date) : 
        d, m, _ = date.split('-')
        d = int(d)
        return f'{d} {m}'

    def H (self, date) : 
        if self.peopleDied : 
            if date < self.firstCases : 
                return np.array([self.h1])
            elif self.firstCases <= date <= self.dataEndDate - 17 :
                return np.array([self.h1, self.h2])
            elif self.dataEndDate - 17 < date <= self.dataEndDate : 
                return np.array([self.h2])
            else :
                return np.array([])
        else :
            if date <= self.dataEndDate : 
                return np.array([self.h2])
            else :
                return np.array([])

    def Z (self, date): 
        if self.peopleDied : 
            if date < self.firstCases : 
                m = self.deaths[date - self.startDate]
                return np.array([m])
            elif self.firstCases <= date <= self.dataEndDate - 17 :
                m = self.deaths[date - self.startDate]
                p = self.P[date - self.firstCases]
                return np.array([m, p])
            elif self.dataEndDate - 17 < date <= self.dataEndDate : 
                p = self.P[date - self.firstCases]
                return np.array([p])
            else :
                return np.array([])
        else : 
            if date <= self.dataEndDate : 
                p = self.P[date - self.firstCases]
                return np.array([p])
            else : 
                return np.array([])

    def R (self, date): 
        if self.peopleDied : 
            if date < self.firstCases : 
                return np.array([1])
            elif self.firstCases <= date <= self.dataEndDate - 17 :
                return np.eye(2)
            elif self.dataEndDate - 17 < date <= self.dataEndDate : 
                return np.array([1])
            else :
                return np.array([])
        else : 
            if date <= self.dataEndDate : 
                return np.array([1])
            else : 
                return np.array([])

    def __call__ (self, T) : 
        endDate = self.startDate + T
        series, variances = extendedKalmanFilter(
                self.model.timeUpdate, self.x0, self.P0, 
                self.Q, self.H, self.R, self.Z, 
                self.startDate, endDate)
        return series, variances

if __name__ == "__main__" : 
    with open('./Data/beta.json') as fd : 
        betas = json.load(fd)
    transportMatrix = np.loadtxt('./Data/transportMatrix.csv', delimiter=',')
    statePop  = [getStatePop(s) for s in Model.STATES]
    mortality = [0.01 * getAgeMortality(s) for s in Model.STATES]
    data = [getData(s) for s in Model.STATES]
    model = Model.IndiaModel(transportMatrix, betas, statePop, mortality, data) 
    for datum, m, nbar,state in zip(data, model.models, statePop, Model.STATES) : 
        E0 = [0, 10, 0]
        A0 = [0, 10, 0]
        I0 = [0, 10, 0]
        nbar[1] -= 30
        x0 = np.array([*(nbar.tolist()), *E0, *A0, *I0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        ks = KalmanSimulator(datum, m, x0)
        series, variances = ks(Date('3 May') - ks.startDate)
        Plot.statePlot(series, variances, state, ks.startDate, 3, datum)
