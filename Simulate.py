from Util import *
from EKF import *
from Date import *
import pickle
import Plot
import json
import Model
from scipy.integrate import odeint
from functools import partial
import numpy as np 
import pandas as pd

def odeSimulator (model, x0, T) :
    dx = partial(model.dx, module=np)
    result = odeint(dx, x0, T)
    return result

def completeSimulator (data, model) : 

    def saveOutput () : 

        def getDeaths () : 
            mortality = data.mortality[idx]
            deadDaily = np.sum(mortality * 0.01 * (x[:,9:12] + x[:,21:24] + x[:,24:27]), axis=1)
            deadDaily = deadDaily[:-17]
            deadDaily = np.concatenate([np.zeros(17), deadDaily])
            deadTotal = np.cumsum(deadDaily)
            return deadDaily, deadTotal

        def getRecovered () : 
            recoveredTotal = np.sum(x[:,27:30], axis=1)
            recoveredDaily = np.insert(np.diff(recoveredTotal), 0 , recoveredTotal[0])
            recoveredDaily = recoveredDaily - deadDaily
            recoveredTotal = np.cumsum(recoveredDaily)
            return recoveredDaily, recoveredTotal

        def getInfected () : 
            infectedActive = np.sum(x[:,3:6] + x[:,15:18] + x[:,6:9] 
                    + x[:,9:12] + x[:,18:21] + x[:,21:24]
                    + x[:,24:27], axis=1)
            infectedDaily = np.insert(np.diff(infectedActive), 0 , infectedActive[0])
            infectedDaily = infectedDaily + recoveredDaily + deadDaily
            return infectedDaily, infectedActive

        with open('xs.pkl', 'wb') as fd : 
            pickle.dump(xs, fd)

        with open('vs.pkl', 'wb') as fd : 
            pickle.dump(vs, fd)

        df1s, df2s = [], []
        for idx, x, v, place in zip(range(model.nPlaces), xs, vs, data.places) : 
            startDate = data.placeData[place]['startDate']
            Plot.statePlot(x, v, place, startDate, 3, data.timeSeries[idx])
            deadDaily, deadTotal = getDeaths()
            recoveredDaily, recoveredTotal = getRecovered()
            infectedDaily, infectedActive = getInfected()
            stateIds = np.ones(tEnd - startDate + 1, dtype = int) * int(idx + 1)

            header1 = ["State Id", "Number of infected (new)", "Number of Death (New)", "Number of Recovery (New)"] 
            header2 = ["State id", "Simulated total infected", "Simulated total death", "Simulated total recovery"]

            df1 = pd.DataFrame(data=[stateIds, infectedDaily, deadDaily, recoveredDaily], index=header1).T
            df2 = pd.DataFrame(data=[stateIds, infectedActive, deadTotal, recoveredTotal], index=header2).T

            datelist = [f'{date.day}/{date.month}/2020' for date in DateIter(startDate, tEnd + 1)]

            df1['Date'] = df2['Date'] = datelist

            df1 = df1[header1]
            df2 = df2[header2]

            df1s.append(df1)
            df2s.append(df2)

        df1 = pd.concat(df1s, ignore_index=True)
        df2 = pd.concat(df2s, ignore_index=True)

        df1.to_csv('sheet1.csv', index=False)
        df2.to_csv('sheet2.csv', index=False)

    def individualPlaceSimulator (place) : 

        def getX0 () : 
            nbar = data.ageBins3[placeIndex]
            nbar[1] -= 30
            x0 = np.zeros(n)
            x0[:3] = nbar
            x0[3:6] = x0[6:9] = x0[9:12] = [0, 10, 0]
            return x0

        def H (date) : 
            if peopleDied : 
                if date < firstCases : 
                    return np.array([h1])
                elif firstCases <= date <= dataEndDate - 17 :
                    return np.array([h1, h2])
                elif dataEndDate - 17 < date <= dataEndDate : 
                    return np.array([h2])
                else :
                    return np.array([])
            else :
                if date <= dataEndDate : 
                    return np.array([h2])
                else :
                    return np.array([])
            
        def Z (date) :
            if peopleDied : 
                if date < firstCases : 
                    m = deaths[date - startDate]
                    return np.array([m])
                elif firstCases <= date <= dataEndDate - 17 :
                    m = deaths[date - startDate]
                    p = P[date - firstCases]
                    return np.array([m, p])
                elif dataEndDate - 17 < date <= dataEndDate : 
                    p = P[date - firstCases]
                    return np.array([p])
                else :
                    return np.array([])
            else : 
                if date <= dataEndDate : 
                    p = P[date - firstCases]
                    return np.array([p])
                else : 
                    return np.array([])

        def R (date) :
            if peopleDied : 
                if date < firstCases : 
                    return np.array([1])
                elif firstCases <= date <= dataEndDate - 17 :
                    return np.eye(2)
                elif dataEndDate - 17 < date <= dataEndDate : 
                    return np.array([1])
                else :
                    return np.array([])
            else : 
                if date <= dataEndDate : 
                    return np.array([1])
                else : 
                    return np.array([])

        placeIndex = data.places.index(place)
        
        startDate = data.placeData[place]['startDate']
        firstCases = data.placeData[place]['firstCase']
        dataEndDate = data.placeData[place]['dataEnd']
        peopleDied = data.placeData[place]['peopleDied']
        deaths = data.placeData[place]['deaths']
        P = data.placeData[place]['P']

        n = 30
        h1, h2 = [0] * n, [0] * n
        h1[9:12] = h1[21:24] = h1[24:27] = data.mortality[placeIndex]
        h2[-6:-3] = [1,1,1]

        P0, Q = np.eye(n), np.eye(n)
        interval = DateIter(startDate, model.models[placeIndex].te)
        xs, vs = dummyKF(model.models[placeIndex].timeUpdate, getX0(), P0, Q, H, R, Z, interval)

        return xs, vs

    tillLockdown = [individualPlaceSimulator(p) for p in data.places]

    x0 = np.hstack([xs[-1] for xs, _ in tillLockdown])
    n = x0.size
    P0 = np.zeros((n, n))
    for i, _ in enumerate(data.places):
        _, vs = tillLockdown[i]
        P0[30*i:30*(i+1), 30*i: 30*(i+1)] = vs[-1]
 
    Q = 0.1 * np.eye(n)
    H = R = Z = lambda t : np.array([])
    tStart = model.lockdownEnd
    tEnd = Date('10 Nov')

    nxs, nvs = dummyKF(model.timeUpdate, x0, P0, Q, H, R, Z, DateIter(tStart, tEnd))

    nvs = [[v[30*i:30*(i+1), 30*i: 30*(i+1)] for i, _ in enumerate(data.places)] for v in nvs]
    nvs = transpose(nvs)

    # Align the new series.
    nxs = nxs.T.reshape((len(data.places), 30, -1))

    # Append the series after the lockdown
    # with before lockdown.
    xs = [np.vstack([tillLockdown[i][0], nxs[i].T[1:,:]]) for i, _ in enumerate(data.places)]
    vs = [tillLockdown[i][1] + nvs[i][1:] for i, _ in enumerate(data.places)]

    saveOutput()

