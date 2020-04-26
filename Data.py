import json
from Date import *
import pandas as pd
import numpy as np
import os 
import os.path as osp
import re

def splitDates(date) : 
    d, m, _ = re.split(r'\-|\/', date)
    return f'{d} {m}'

def getActiveCases (data) : 
    totalConfirmed = data['Total Cases']
    totalRecovered = data['Total Recovered']
    totalDeceased  = data['Total Dead']
    return (totalConfirmed - totalRecovered - totalDeceased).to_numpy()

def getAgeBins (dirPath, place) : 
    fname = place + '.csv'
    path = osp.join(dirPath, fname)
    return np.loadtxt(path, delimiter=',', usecols=(1))

def getAgeMortality (ageWiseMortalityPath, ageBins10Path, place) : 
    ageWise = np.loadtxt(ageWiseMortalityPath, delimiter=',', usecols=(1))
    popHistogram = getAgeBins(ageBins10Path, place)

    till60 = popHistogram[:ageWise.size-1]
    after60 = popHistogram[ageWise.size-1:].sum()
    popHistogram = np.hstack((till60, after60))

    deaths = popHistogram * ageWise

    less20 = deaths[:2].sum() / popHistogram[:2].sum()
    between20_60 = deaths[2:6].sum() / popHistogram[2:6].sum()
    above60 = deaths[6:].sum() / popHistogram[6:].sum()

    mortality = np.array([less20, between20_60, above60])
    return mortality
    
def getTimeSeriesData (dirPath, place) : 
    fname = place + '.csv'
    path = osp.join(dirPath, fname)
    return pd.read_csv(path)

class Data () : 
    """
    Handle all the data to run the simulations.
    """
    
    def __init__ (self, configFilePath) : 
        with open(configFilePath) as fd : 
            self.config = json.load(fd)

        transportMatrix = pd.read_csv(self.config['transportMatrix'])

        self.contactHome = np.loadtxt(self.config['contactHome'], delimiter=',')
        self.contactTotal = np.loadtxt(self.config['contactTotal'], delimiter=',')

        self.transportMatrix = transportMatrix.iloc[:, 1:].to_numpy()
        self.places = transportMatrix.iloc[:, 0].tolist()

        self.timeSeries = [getTimeSeriesData(self.config['timeSeries'], p) for p in self.places]
        self.ageBins3 = [getAgeBins(self.config['ageBins3'], p) for p in self.places]
        self.mortality = [getAgeMortality(self.config['ageMortality'], self.config['ageBins10'], p) for p in self.places]

        self.placeData = dict()

        for i, p in enumerate(self.places) :
            data = self.timeSeries[i]
            dates = data['Date'].map(splitDates)
            firstCase = Date(dates.iloc[0]) 
            dataEnd = Date(dates.iloc[-1])
            peopleDied = dates[data['Total Dead'] > 0].size > 0 
            deaths = None
            firstDeath = None

            if peopleDied : 
                firstDeath = Date(dates[data['Total Dead'] > 0].iloc[0])
                deaths = data['Daily Dead'][data['Total Dead'] > 0].to_numpy()
                startDate = firstDeath - 17
            else : 
                startDate = firstCase

            P = getActiveCases(data)

            self.placeData[p] = {
                'dates' : dates,
                'firstCase' : firstCase,
                'dataEnd' : dataEnd,
                'peopleDied' : peopleDied,
                'deaths' : deaths,
                'firstDeath' : firstDeath,
                'startDate' : startDate,
                'P' : P
            }

