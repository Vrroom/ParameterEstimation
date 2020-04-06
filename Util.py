import more_itertools
import csv
import numpy as np

def getInfectedAndRecovered(csvFile):
    with open(csvFile) as fd :
        reader = csv.reader(fd)
        confirmedOffset, recoveredOffset, deadOffset = 0, 0, 0
        i, r = [], []
        for row in reader:
            confirmed = int(row[2]) - confirmedOffset
            recovered = int(row[4]) - recoveredOffset
            dead = int(row[6]) - deadOffset
            r.append(recovered + dead)
            i.append(confirmed - r[-1])
    return np.array([i, r])

def sortAndFlattenDict(d) : 
    return list(unzip(sorted(d.items()))[1])
