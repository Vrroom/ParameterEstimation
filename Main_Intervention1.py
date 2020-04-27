from Simulate import completeSimulator
from Model import ConnectedSpaxire
from Data import Data 
from Date import *
from Util import climbFn

topStates = ["MIZORAM","KERALA","PUNJAB","TAMILNADU","GUJARAT","HIMACHALPRADESH","LAKSHADWEEP","MAHARASHTRA","KARNATAKA","ANDHRAPRADESH","WESTBENGAL","JAMMU&KASHMIR","LADAK"]
midStates = ["TELANGANA", "CHANDIGARH", "MANIPUR", "MEGHALAYA", "SIKKIM", "CHHATTISGARH", "ARUNACHALPRADESH", "GOA", "NCTOFDELHI", "JHARKHAND", "HARYANA", "ANDAMAN&NICOBAR", "PUDUCHERRY"]
bottomStates = ["TRIPURA", "UTTARAKHAND", "ASSAM", "NAGALAND", "DAMAN&DIU", "MADHYAPRADESH", "ODISHA", "BIHAR", "RAJASTHAN", "DADRA&NAGARHAVELI", "UTTARPRADESH"]
def intervention(model, data):
	increaseTestingStartDate = Date('26 Apr')
	increaseTestingEndDate = Date('3 May')
	keralaTestingFraction = data.testingFractions['KERALA']
	newTestingFractions = data.testingFractions
	for state in topStates:
		assert state in newTestingFractions.keys(), "State code is inccorect: " + state
		newTestingFractions[state] = \
			[lambda x: climbFn(x, increaseTestingStartDate, increaseTestingEndDate, stateTestingFraction, 1.0 * keralaTestingFraction) \
				if keralaTestingFraction > stateTestingFraction else stateTestingFraction \
				for stateTestingFraction, keralaTestingFraction in zip(data.testingFractions[state], keralaTestingFraction)]
	
	for state in midStates:
		assert state in newTestingFractions.keys(), "State code is inccorect: " + state
		newTestingFractions[state] = \
			[lambda x: climbFn(x, increaseTestingStartDate, increaseTestingEndDate, stateTestingFraction, 0.5 * keralaTestingFraction) \
				if keralaTestingFraction > stateTestingFraction else stateTestingFraction \
				for stateTestingFraction, keralaTestingFraction in zip(data.testingFractions[state], keralaTestingFraction)]
	
	for state in bottomStates:
		assert state in newTestingFractions.keys(), "State code is inccorect: " + state
		newTestingFractions[state] = \
			[lambda x: climbFn(x, increaseTestingStartDate, increaseTestingEndDate, stateTestingFraction, 0.25 * keralaTestingFraction) \
				if keralaTestingFraction > stateTestingFraction else stateTestingFraction \
				for stateTestingFraction, keralaTestingFraction in zip(data.testingFractions[state], keralaTestingFraction)]

if __name__ == "__main__"  : 
    data = Data('./config.json')
    model = ConnectedSpaxire(data)
    completeSimulator(data, model, changeOnLockdown = intervention) 
