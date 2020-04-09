import more_itertools
import contextlib
import pandas
import sys
import numpy as np
from itertools import product
import os

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def getInfectedAndRecovered(csvFile):
    data = pandas.read_csv(csvFile)
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
