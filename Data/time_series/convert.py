import numpy as np

file = open('rawData.csv')
lines = file.readlines()
file.close()
lines = [line.replace('\n', '').replace(',,', ',0,').split(',') for line in lines if len(line) > 0][:-3]

finalData = {}
numDates = (len(lines) - 1) // 3

for state in lines[0][2:]:
	finalData[state] = [[None for _ in range(7)] for _ in range(numDates)]
	# for each row, column: 0 -> date
	#						1 -> daily confirmed
	#						2 -> total confirmed
	#						3 -> daily dead
	#						4 -> total dead
	#						5 -> daily recovered
	#						6 -> total recovered

for i, line in enumerate(lines[1:]):
	entryNumber = i // 3
	for j, entry in enumerate(line):
		if j < 2: continue
		currentState = lines[0][j]
		if line[1] == 'Confirmed':
			finalData[currentState][entryNumber][1] = int(entry)
			finalData[currentState][entryNumber][0] = line[0]
		elif line[1] == 'Deceased':
			finalData[currentState][entryNumber][3] = int(entry)
		elif line[1] == 'Recovered':
			finalData[currentState][entryNumber][5] = int(entry)
		else:
			print("Unexpected entry:")
			print(line)


# Now compute the total from the daily
def populateTotal(data):
	totalConfirmed, totalDead, totalRecovered = 0, 0, 0
	for i in range(len(data)):
		totalConfirmed += data[i][1]
		totalDead += data[i][3]
		totalRecovered += data[i][5]
		data[i][2] = totalConfirmed
		data[i][4] = totalDead
		data[i][6] = totalRecovered
	return data


# Write to CSV
for state in finalData:
	file = open(state + '.csv', 'w')
	finalData[state] = populateTotal(finalData[state])
	file.write(','.join(['Date', 'New Cases', 'Total Cases', 'Daily Dead', 'Total Dead', 'Daily Recovered', 'Total Recovered']))
	file.write('\n')
	for x in finalData[state]:
		file.write(','.join([str(y) for y in x]))
		file.write('\n')
	file.close()

