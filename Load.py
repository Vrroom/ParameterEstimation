import pickle

with open('derivMap.pkl', 'rb') as fd : 
    derivMap = pickle.load(fd)
with open('par.pkl', 'rb') as fd : 
    par = pickle.load(fd)
with open('var.pkl', 'rb') as fd : 
    var = pickle.load(fd)
with open('map.pkl', 'rb') as fd : 
    m = pickle.load(fd)

import pdb
pdb.set_trace()
print(m)
