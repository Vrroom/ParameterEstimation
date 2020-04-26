from Simulate import completeSimulator
from Model import ConnectedSpaxire
from Data import Data 

if __name__ == "__main__"  : 
    data = Data('./config.json')
    model = ConnectedSpaxire(data)
    completeSimulator(data, model) 
