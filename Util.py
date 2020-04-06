import more_itertools
import pandas
import numpy as np
from itertools import product

def getInfectedAndRecovered(csvFile):
    data = pandas.read_csv()
    confirmed = data['Total Cases']
    recovered = data['Total Recoveries']
    dead      = data['Total Deaths']
    r = recovered + dead
    i = confirmed - r
    return pandas.concat((i, r), axis=1).to_numpy()

def sortAndFlattenDict(d) : 
    return list(unzip(sorted(d.items()))[1])

def dictProduct (d) : 
    return map(dict, product(*map(lambda x : product([x[0]], x[1]), d.items())))
