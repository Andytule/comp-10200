#!/usr/bin/env python
# coding: utf-8

# In[9]:


"""
I, Andy Le, student number 000805099, certify that all code submitted is my
own work; that I have not copied it from any other source. I also certify that I
have not allowed my work to be copied by others.
"""

import numpy as np
import time

names = set()
times = []

playerNumber = int(input("How many players? "))
timeInterval = int(input("How many time intervals? ")) + 1

print("\nEnter 5 player names")

while(0 < playerNumber):
    nameInput = input().lower().replace(" ", "").capitalize()
    if (not nameInput):
        print("Name can't be empty")
    elif (nameInput in names):
        print("Name already entered")
    else:
        names.add(nameInput)
        playerNumber -= 1
        
npNames = np.array(list(names))
np.random.shuffle(npNames)

print()

for name in npNames:
    playerTimes = []
    print(name + "'s turn. Press enter " + str(timeInterval) + " times quickly.")
    input()
    for i in range(1, timeInterval):
        time1 = time.time()
        input()
        time2 = time.time()
        playerTimes.append(round((time2 - time1), 3))
    times.append(playerTimes)

npTimes = np.array(times)

# Sorting arrays alphabetically
sort = npNames.argsort()
npTimes = npTimes[sort]
npNames = npNames[sort]

# Mean of times and round to 3 decimal places
mean = npTimes.mean(axis = 1)
mean = np.around(mean, decimals = 3)

# Fastest Average Time
minAvgTime = round(np.mean(npTimes, axis = 1).min(), 3)
minAvgName = npNames[ np.mean(npTimes, axis = 1).argmin() ]

# Slowest Average Time
maxAvgTime = round(np.mean(npTimes, axis = 1).max(), 3)
maxAvgName = npNames[ np.mean(npTimes, axis = 1).argmax() ]

# Fastest Single Time
minOneTime = round(np.min(npTimes, axis = 1).min(), 3)
minOneName = npNames[ np.min(npTimes, axis = 1).argmin() ] 

# Slowest Single Time
maxOneTime = round(np.max(npTimes, axis = 1).max(), 3)
maxOneName = npNames[ np.max(npTimes, axis = 1).argmax() ] 

# Output
print("Names " + str(npNames))
print("Mean times: " + str(mean))
print("Fastest Average Time: " + str(minAvgTime) + " by " + minAvgName)
print("Slowest Average Time: " + str(maxAvgTime) + " by " + maxAvgName)
print("Fastest Single Time: " + str(minOneTime) + " by " + minOneName)
print("Slowest Single Time: " + str(maxOneTime) + " by " + maxOneName)
print()
print(npNames)
print(npTimes)


# In[ ]:





# In[ ]:





# In[ ]:




