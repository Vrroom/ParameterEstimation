from itertools import product
from functools import partial
import time
from Simulate import *
from Model import * 
from Search import *
from multiprocessing import Process, Pipe, cpu_count
import math

def gridSearch (ivRanges, paramRanges, groundTruth, lossFunction, T) :
    lo, hi = T.min(), T.max()
    samples = 5
    T_ = np.linspace(lo, hi, (hi - lo) * samples)
    startIdx = groundTruth[groundTruth['Date'] == '20 Mar'].index[0] 
    deaths = groundTruth['New Deaths'][startIdx:].to_numpy()
    nDays = deaths.size
    minLoss = math.inf
    minx0, minParams = None, None
    for x0, params in product(product(*ivRanges), dictProduct(paramRanges)) :
        model = Sixer(x0, params)
        result = simulator(model, T_)
        infections = result[:, 2][::samples]
        tested = result[:, -1][::samples]
        deathEstimate = 0.02 * (infections + tested)
        deathEstimate = deathEstimate[:nDays]
        loss = lossFunction(deaths, deathEstimate)
        if loss < minLoss : 
            minx0 = x0
            minParams = params
            minLoss = loss

    return minx0, minParams

def worker (conn, groundTruth, lossFunction, T) :

    # Do pre-processing of the deaths.
    lo, hi = T.min(), T.max()
    samples = 5
    T_ = np.linspace(lo, hi, (hi - lo) * samples)
    startIdx = groundTruth[groundTruth['Date'] == '20 Mar'].index[0] 
    deaths = groundTruth['New Deaths'][startIdx:].to_numpy()
    nDays = deaths.size

    while True : 
        item = conn.recv() 
        if item == 'e' : 
            conn.send('e')
            break
        else : 
            point, minLoss = item
            x0, params = point
            N = params['N']
            x0 = [N - sum(x0), *x0]
            model = Sixer(x0, params)
            result = simulator(model, T_)
            infections = result[:, 2][::samples]
            tested = result[:, -1][::samples]
            deathEstimate = 0.02 * (infections + tested)
            deathEstimate = deathEstimate[:nDays]
            loss = lossFunction(deaths, deathEstimate)
            if loss < minLoss: 
                conn.send((point, loss))
            else : 
                conn.send('n')

def parallelGridSearch (ivRanges, paramRanges, groundTruth, lossFunction, T) :
    totalChildren = cpu_count()

    # Setup duplex connections between parent and child.
    connections = [Pipe() for _ in range(totalChildren)]

    # Setup Child Processes
    processes = [
        Process(target=worker, args=(conn, groundTruth, lossFunction, T))
        for _, conn in connections
    ]  

    # Start Child Processes
    for p in processes : 
        p.start() 

    doneChildren =  0

    # Grid point generator
    points = product(product(*ivRanges), dictProduct(paramRanges)) 

    # Best grid point
    minLoss = math.inf
    minX0, minParams = None, None

    # Send initial data
    for p, _ in connections : 
        p.send((next(points), minLoss))

    nProcessed = 0
    power = 1
    st = time.time()

    while doneChildren < totalChildren : 

        # Poll all connections and 
        # send stuff if they are available.
        for conn, _ in connections : 
            hasSomething = conn.poll()
            if hasSomething : 
                obj = conn.recv() 
                if obj == 'e' :
                    # This signals that the child
                    # is done.
                    doneChildren += 1
                elif obj != 'n' : 
                    # This signals that the child
                    # found an improving grid point.
                    point, loss = obj
                    if loss < minLoss : 
                        minX0, minParams = point
                        minLoss = loss

                # Sentinel to guard against the 
                # possibility of querying an empty generator.
                point = next(points, False)

                nProcessed += 1
                if nProcessed % (10 ** power) == 0 : 
                    print(nProcessed, time.time() - st)
                    power += 1

                if point : 
                    conn.send((point, minLoss))
                else : 
                    # If no more points are left, signal
                    # to the child.
                    conn.send('e')

    # Join all child processes.
    for p in processes : 
        p.join()

    return minX0, minParams, minLoss
