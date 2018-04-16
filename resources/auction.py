from .globalv import * 
from .generalFunctions import *
from .optimizeMILP import *
import numpy as np


def runDescendingAuction(elements, linklist, destinations, storedtasks, newtasks, time, federates): 
	global storagepenalty, epsilon, linkcapacity, elementcapacity, firstprice
	
	avgsharecost = np.mean([f.sharelinkcost for f in federates])
	avgusecost = np.mean([f.uselinkcost for f in federates])
	
	lastlinklist = []
	totalvaluelist = []
	lastvalue = 0
	lastlinklist = templinklist = [li for li in linklist if li.owner.sharelinkcost < firstprice]
	for price in range(firstprice, -1, -dpri): 
		templinklist = [li for li in lastlinklist if li.owner.sharelinkcost < price]
		# print("length of templinklist:", len(templinklist))
		if not templinklist: 
			totalvaluelist.append(lastvalue)
			continue 
			
		# if price != firstprice and len(templinklist) == len(lastlinklist):
		# 	totalvaluelist.append(lastvalue)
		# 	continue 
			
		# print("Avg share and use cost, length of links and new links:", avgsharecost, avgusecost, price, len(linklist), len(lastlinklist))
		storedtasks, edges2 = optimizeAuction(elements = elements, linklist = lastlinklist, destinations = elementnames[-2:], storedtasks = storedtasks, newtasks = newtasks, time = time, federates = federates, price = price)
		lastvalue = sum([f.cash for f in federates])
		totalvaluelist.append(lastvalue) 
		# print(templinklist)
		# print(edges2)
		lastlinklist = [l for l in templinklist if (l.source.name, l.destin.name) not in edges2]
		# print("New length and selected links:", len(templinklist), len(lastlinklist), len(edges2))
	
	# print("totalvaluelist:", totalvaluelist)
	return totalvaluelist
	