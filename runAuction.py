"""
Copyright 2018, Abbas Ehsanfar, Stevens Institute of Technology
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from resources.classes import *
from resources.globalv import * 
from collections import defaultdict, Counter, namedtuple
from itertools import product 
import pickle
import os
import random
import hashlib
from resources.optimizeMILP import optimizeMILP
from resources.optimizeSLSQP import optimizeSLSQP
from multiprocessing import Process, Manager
import argparse
import os
import copy

dir_topologies = 'topologies_new/' 
dir_simulations = 'simulations/'
random.seed(seed)

def findsolution(nettopObj, solutionObj, priceDict): 
	global milpObjDict, Objects
	if solutionObj.hashid in milpObjDict[nettopObj.hashid]:
		return milpObjDict[nettopObj.hashid][solutionObj.hashid]
		
	edgePriceDict = {e: priceDict[Objects.elfedDict[e[1]]][0] for e in nettopObj.edges}
	# findsolution(nettopObj, edgePriceDict, solObj)
	# federatenames = nettopObj.federates
	# federates = [Federate(name = f, cash = 0, sharelinkcost = sharelinkcost, uselinkcost = uselinkcost) for f in set(federatenames)]
	# federateDict = {f.name: f for f in federates}
	# elements = [Element(name = e, capacity=elementcapacity, size = 0, owner = federateDict[f]) for (e,f) in zip(nettopObj.elements, federatenames)]
	# elementDict = {e.name: e for e in elements}
	# Objects.sources = [e for e in elements if e.name not in nettopObj.destinations]
	# # Objects.sources = nettopObj.Objects.sources
	# linklist = [Link(source = elementDict[e1], destin = elementDict[e2], capacity = linkcapacity, size = 0, owner = elementDict[e2].owner) for (e1, e2) in nettopObj.edges]
	# time = 0
	# newtasks = [Task(id = id + n, element=s, lastelement=s, size=size, value=value, expiration=time + 5, init=time, active=True, penalty=penalty) for n, s in enumerate(Objects.sources)]
	# for sharelinkcost, uselinkcost in [(400, 801)]:
	# for f in federates: 
	# 	f.cash = 0 
	# 	f.sharelinkcost = sharelinkcost
	# 	f.uselinkcost = uselinkcost
	# netObjectsorted(nettopObj.elements, key = lambda x: len(x)))
	solutionObj = optimizeMILP(elements = Objects.elements, linklist = Objects.linklist, destinations = nettopObj.destinations, 
		storedtasks = [], newtasks = Objects.newtasks, time = time, federates = Objects.federates, edgePriceDict = edgePriceDict, 
		solutionObj = solutionObj)
	
	milpObjDict[nettopObj.hashid][solutionObj.hashid] = solutionObj
	return solutionObj
	# totalvalue = sum(solutionObj.fedValDict.values())
			
# def dfs(edges, Objects.sources, destinations): 
# 	global linkcapacity
# 	edgecounter = Counter()
# 	graph = {e[0]: set([]) for e in edges}
	
# 	for e in edges: 
# 		graph[e[0]].add(e[1])
	
# 	for source in Objects.sources:
# 		stack = [(source, [source])]
		
# 		while stack: 
# 			vertex, path = stack.pop()
# 			if vertex not in graph: 
# 				continue
# 			nextlist = graph[vertex]
			
# 			for nextv in nextlist: 
# 				if nextv in destinations: 
# 					yield path + [nextv]
# 				elif nextv not in path: 
# 					stack.append((nextv, path + [nextv]))


# def isBundle(pathlist, Objects.elfedDict):
# 	global linkcapacity, epsilon, sourcecostlimitDict, edgePriceDict
# 	edgecounter = Counter()
# 	sumcostconst = []
# 	for path in pathlist: 
# 		source = path[0]
# 		sourceFed = Objects.elfedDict[source]
# 		tempedgelist = list(zip(path, path[1:]))
# 		edgecounter += Counter(tempedgelist)
# 		pathedgecost = [edgePriceDict[e] if sourceFed != Objects.elfedDict[e[1]] else epsilon for e in tempedgelist]
# 		sumedgecost = sum(pathedgecost)
# 		sumcostconst.append(sourcecostlimitDict[path[0]]>=sumedgecost)
	
# 	try: 
# 		return max(edgecounter.values())<= linkcapacity and all(sumcostconst)
# 	except: 
# 		return False

def searchPriceBinary(nettopObj, fedPriceLowerDict, fedPriceUpperDict, fedBidDict): 
	global milpObjDict
	nf = len(fedPriceLowerDict)
	lowerDict = {f: int(10*(v//10)) for f, v in fedPriceLowerDict.items()}
	upperDict = {f: int(10*(v//10)) for f, v in fedPriceUpperDict.items()}
	# lowerCostDict = {f: (lowerDict[f], federateDict[f].uselinkcost) for f in lowerDict}
	# lowerObj = MILPSolution(nettopObj.hashid, time, fedPriceDict = {f: (lowerDict[f], fedPriceDict[f][1]) for f in lowerDict}, fedValDict = {f: 0 for f in lowerDict.keys()}, edgelist = [])
	# upperObj = MILPSolution(nettopObj.hashid, time, fedPriceDict = {f: (upperDict[f], fedPriceDict[f][1]) for f in upperDict}, fedValDict = {f: 0 for f in upperDict.keys()}, edgelist = [])
	lowerObj = LiteSolution(fedBidDict = {f: (lowerDict[f], fedBidDict[f][1]) for f in lowerDict}, nettopObj = nettopObj)
	upperObj = LiteSolution(fedBidDict = {f: (upperDict[f], fedBidDict[f][1]) for f in upperDict}, nettopObj = nettopObj)
	if lowerObj.hashid not in milpObjDict[nettopObj.hashid]: 
		tempDict = {f: (v, fedBidDict[f][1]) for f, v in lowerDict.items()}
		lowerObj = findsolution(nettopObj, lowerObj, tempDict)
	else: 
		lowerObj = milpObjDict[nettopObj.hashid][lowerObj.hashid]
	
	if upperObj.hashid not in milpObjDict[nettopObj.hashid]:
		tempDict = {f: (v, fedBidDict[f][1]) for f, v in upperDict.items()}
		upperObj = findsolution(nettopObj, upperObj, tempDict)
	else: 
		upperObj = milpObjDict[nettopObj.hashid][upperObj.hashid]
				
	equalDict = {f: lowerDict[f] for f in lowerDict if lowerDict[f] == upperDict[f]}
	# for f in equalDict: 
	# 	del lowerDict[f]
	# 	del upperDict[f]
		
	oldDict = {}
	midDict = {}
	i = 0 
	while i<20: 
		# oldDict = midDict.copy()
		midDict = {f: equalDict[f] if f in equalDict else 10*((lowerDict[f] + upperDict[f])//20) for f in lowerDict}
		tempDict = {f: (v, fedBidDict[f][1]) for f, v in midDict.items()}
		# midObj = MILPSolution(nettopObj.hashid, time, fedPriceDict = tempDict, fedValDict = {f: 0 for f in midDict.keys()}, edgelist = [])
		midObj = LiteSolution(fedBidDict = tempDict, nettopObj = nettopObj)
		if midObj.hashid not in milpObjDict[nettopObj.hashid]:	
			midObj = findsolution(nettopObj, midObj, tempDict)
		else:
			midObj = milpObjDict[nettopObj.hashid][midObj.hashid]
		
		assert midObj.totalvalue <= lowerObj.totalvalue
		if midObj.totalvalue < lowerObj.totalvalue: 
			upperDict = midDict.copy()
			upperObj = copy.deepcopy(midObj)
		
		else: 
			lowerObj = copy.deepcopy(midObj)
			lowerDict = midDict.copy()
				
		if oldDict == midDict:
			break
		
		oldDict = midDict.copy()
		i +=1 
	
	if i == 20: 
		midObj = lowerObj		
	
	return (midObj, midDict)
	
	
def findPriceBoundary(nettopObj, bundle, fedBidDict):
	global Objects
	sourcecostlimitDict= {s.name: fedBidDict[Objects.elfedDict[s.name]][1] for s in Objects.sources}
	def findLowerUpperDict(edgelist, edgeMinDict, source, sourcecostlimitDict, edgeLowerpriceDict): 
		assert sourcecostlimitDict[source] - sum([edgeMinDict[e] if e in edgeMinDict else epsilon for e in edgelist])>=0 
		lowerpriceDict = {}
		upperpriceDict = {}
		pathMinDict = {e: p for e,p in edgeMinDict.items() if e in edgelist}
		sortedprice = sorted(list(set(list(pathMinDict.values()))), reverse = True)
		upperprice = lowerprice = sortedprice[0]
		for lowerprice in sortedprice: 			
			if sourcecostlimitDict[source] -  sum([max(pathMinDict[e], lowerprice) for e in edgelist if e in pathMinDict]) - epsilon * (len(edgelist) - len(pathMinDict))>0: 
				break
			upperprice = lowerprice
		
		for e, minprice in pathMinDict.items(): 
			lowerpriceDict[e] = max(minprice, lowerprice)
			# upperpriceDict[e] = max(minprice, upperprice)
		lowerprice = (sourcecostlimitDict[source] - epsilon * (len(edgelist) - len(pathMinDict)) - sum([pathMinDict[e] for e in pathMinDict if pathMinDict[e] > lowerprice]))/len([v for v in pathMinDict.values() if v <= lowerprice])
		# lowerprice += deltaprice
		for e, v in pathMinDict.items():
			if e in edgeLowerpriceDict:
				edgeLowerpriceDict[e] = min(edgeLowerpriceDict[e], max(v, lowerprice))
			else: 
				edgeLowerpriceDict[e] = max(v, lowerprice)
		
	edgeMinDictTotal = {}
	pathUpperDict = {}
	edgeLowerpriceDict = defaultdict(float)
	edgeUpperpriceDict = defaultdict(float)
	for edgelist in bundle: 
		edgeMinDict = {}
		source = edgelist[0][0]
		sourcefed = Objects.elfedDict[edgelist[0][0]]
		# tempedgelist = list(zip(path, path[1:]))
		for edge in edgelist: 
			edgefed = Objects.elfedDict[edge[1]]
			if sourcefed != edgefed: 
				edgeMinDict[edge] = fedBidDict[edgefed][0]
				edgeMinDictTotal[edge] = edgeMinDict[edge]
			
			else: 
				edgeMinDict[edge] = epsilon
		
		
		findLowerUpperDict(edgelist, edgeMinDict, source, sourcecostlimitDict, edgeLowerpriceDict)
	
	
	
	tempfedpricelistDict = defaultdict(list)
	for (e1, e2), p in edgeLowerpriceDict.items(): 
		fedname = Objects.elfedDict[e2]	
		tempfedpricelistDict[fedname].append(p)
	
	fedPriceLowerDict = {f: max(min(l), fedBidDict[f][0]) for f, l in tempfedpricelistDict.items()}
	fedPriceUpperDict = {f: max([lim for f1, lim in sourcecostlimitDict.items() if f1!=f]+[fedPriceLowerDict[f]]) for f in fedPriceLowerDict}
	
	for f in set(nettopObj.federates): 
		if f not in fedPriceLowerDict: 
			fedPriceLowerDict[f] = fedBidDict[f][0] 
			fedPriceUpperDict[f] = max(list(sourcecostlimitDict.values())+[fedPriceLowerDict[f]])
	# upperprice = min(sourcecostlimitDict.values())
	return (fedPriceLowerDict, fedPriceUpperDict)

def buildObjects(nettopObj): 	
	global time, Objects
	Objects.federatenames = list(set(nettopObj.federates))
	Objects.numfederates = len(Objects.federatenames)
	# fedPriceDict = {fname: (sharelinkcost - int(fname[-1])*100, uselinkcost - int(fname[-1])*100) for fname in federatenames}
	Objects.federates = [Federate(name = f, cash = 0) for f in set(Objects.federatenames)]
	Objects.federateDict = {f.name: f for f in Objects.federates}
	
	Objects.elements = [Element(name = e, capacity=elementcapacity, size = 0, owner = Objects.federateDict[f]) for (e,f) in zip(nettopObj.elements, nettopObj.federates)]
	Objects.elementDict = {e.name: e for e in Objects.elements}
	Objects.sources = [e for e in Objects.elements if e.name not in nettopObj.destinations]
	# Objects.sources = nettopObj.Objects.sources
	Objects.linklist = [Link(source = Objects.elementDict[e1], destin = Objects.elementDict[e2], capacity = linkcapacity, size = 0, owner = Objects.elementDict[e2].owner) for (e1, e2) in nettopObj.edges]
	Objects.newtasks = [Task(id = id + n, element=s, lastelement=s, size=size, value=value, expiration=time + 5, init=time, active=True, penalty=penalty) for n, s in enumerate(Objects.sources)]
	Objects.elfedDict = {e: f for e, f in zip(nettopObj.elements, nettopObj.federates)}
	Objects.centPriceDict = {f: (epsilon, 1000) for f in Objects.federatenames}
	Objects.indPriceDict = {f: (1000, 0) for f in Objects.federatenames}

	return Objects

		
def findbundles(nettopObj, solutionObj): 	
	global Objects, sourcecostlimitDict
	edgelist = [e for l in solutionObj.sourceEdgeDict.values() for e in l]
	# edgelist = solutionObj.
		# for f in federates: 
	# 	f.cash = 0 
	# 	f.sharelinkcost = sharelinkcost
	# 	f.uselinkcost = uselinkcost
	# sourcecostlimitDict= {s.name: federateDict[Objects.elfedDict[s.name]].uselinkcost for s in Objects.sources}
	# edgePriceDict = {e: federateDict[Objects.elfedDict[e[1]]].sharelinkcost for e in nettopObj.edges}
	# findsolution(nettopObj, solutionObj, {f: tup[0] for f, tup in fedPriceDict.items()})	
	# soledges = solutionObj.edgelist
	# for soledges in solutionedges:
	sourcePathDict = defaultdict(list)
	# paths = list(dfs(edgelist, nettopObj.Objects.sources, nettopObj.destinations))
	pathedgelist = list(solutionObj.sourceEdgeDict.values()) 
	return pathedgelist
	# for edgelist in pathedgelist: 
	# 	sourcePathDict[edgelist[0][0]].append(edgelist)
	
	# permulations = product(*[list(range(len(pl))) for pl in sourcePathDict.values()])
	# for perm in permulations: 
	# 	pathlist = [pl[i] for i, pl in zip(perm, sourcePathDict.values())]
	# 	if isBundle(pathlist, Objects.elfedDict):
	# 		return pathlist	
	# pathDict = [p[0]]
	
# def testdfs(): 
# 	with open(dir_topologies + 'hashNetworkDict_elements10_federates2_density7_top10.p', 'rb') as infile: 
# 		selected = pickle.load(infile)[0]
		
# 	solutionedges = findsolution(selected, )
# 	for soledges in solutionedges:
# 		paths = list(dfs(soledges, selected.Objects.sources, selected.destinations))
# 		# for p in paths: 

# def testpathlist(pathedgelist, solutionObj, taskvalue = 1000): 
# 	global Objects.elfedDict, epsilon
# 	priceDict = {f: tup[0] for f, tup in solutionObj.fedPriceDict.items()}
# 	linkcostDict = defaultdict(int)
# 	linkvalueDict = defaultdict(int)
	
# 	for edgelist in pathedgelist:
# 		sourcefed = Objects.elfedDict[edgelist[0][0]]
# 		for e in edgelist: 
# 			linkcostDict[sourcefed] += priceDict[Objects.elfedDict[e[1]]] if Objects.elfedDict[e[1]] != sourcefed else epsilon
# 			linkvalueDict[Objects.elfedDict[e[1]]] += priceDict[Objects.elfedDict[e[1]]] if Objects.elfedDict[e[1]] != sourcefed else epsilon - epsilon

# 	for name in solutionObj.fedValDict:
# 		fedpathlist = [Objects.elfedDict[edgelist[0][0]] for edgelist in pathedgelist if Objects.elfedDict[edgelist[0][0]] == name]
# 		fedtaskvalue = taskvalue * len([Objects.elfedDict[edgelist[0][0]] for edgelist in pathedgelist if Objects.elfedDict[edgelist[0][0]] == name]) 
# 		fedvalue = solutionObj.fedValDict[name]
# 	# 	fedvalue = solutionObj.fedValDict[name]
# 	# 	modifiedlist = []
# 	# 	linkcost = 0
# 	# 	for edgelist in pathedgelist: 
# 	# 		pathcostlist = [priceDict[Objects.elfedDict[e[1]]] if Objects.elfedDict[e[1]] != pathfed else epsilon for e in edgelist]
# 	# 		if Objects.elfedDict[edgelist[0][0]] == name:
# 	# 			linkcost += sum(pathcostlist)
# 	# 		else: 
# 	# 			linkvalue += 
# 	# 		# edgelist = list(zip(path, path[1:]))
# 	# 		pathfed = Objects.elfedDict[edgelist[0][0]]
# 	# 		linkcost += 			
# 			# if pathcost <= taskvalue: 
# 			# 	modifiedlist.append(edgelist)
		
# 		# fedpathlist = [el for el in modifiedlist if Objects.elfedDict[el[0][0]] == name]            
# 		# fedtaskvalue = taskvalue * len(fedpathlist)
# 		# linkcost = 0
# 		# for el in modifiedlist: 
# 		# 	if Objects.elfedDict[el[0][0]] == name: 
# 		# 		for e in el: 
# 		# 			if 
# 		# 	for e in el: 
				
			
# 		# outedgelist = [e for e in [e for l in modifiedlist for e in l] if Objects.elfedDict[e[1] != name]
# 		# tempotherpaths =  [el for el in modifiedlist if Objects.elfedDict[el[0][0]] != name]
# 		# inedgelist =  [e for e in [e for l in tempotherpaths for e in l] if Objects.elfedDict[e[1]] == name]

# 		# linkcost = sum([priceDict[Objects.elfedDict[e[1]]] for e in outedgelist])
# 		# linkvalue = sum([priceDict[name] for e in inedgelist])
	

def runAuctionProc(baseObj, nettopObj, fedBidDictList, proc):
	global sol_filename, milp_filename, Objects, milpObjDict
	
	proc_sol_filename = sol_filename.replace('.p','')  + '_proc%s.p'%str(proc).zfill(3)
	proc_milp_filename = milp_filename.replace('.p','')  + '_proc%s.p'%str(proc).zfill(3)
	
	if os.path.isfile(proc_sol_filename): 
	    with open(proc_sol_filename, 'rb') as infile:
	        procSolObjDict = pickle.load(infile)	    	
	else: 
		procSolObjDict = {}
	
	nf = len(set(nettopObj.federates))
	for k, fedBidDict in enumerate(fedBidDictList):
		# try:
			solutionObj = MILPSolution(nettopObj.hashid, time, fedBidDict = fedBidDict, edgelist = [])
			solhashid = solutionObj.hashid
			if solhashid in solutionObjDict or solhashid in procSolObjDict:
				print(solhashid, " exists")
				continue 

			solutionObj.centrlaizedValues = baseObj['centrlaizedValues'].copy()
			solutionObj.independentValues = baseObj['independentValues'].copy()			
			solutionObj.centralizedPathlist = baseObj['centralizedPathlist'].copy()
			solutionObj.independentPathlist = baseObj['independentPathlist'].copy()
			pathedgelist_cent = baseObj['pathedgelist_cent']
			
			tempObj = LiteSolution(fedBidDict, nettopObj)
			# sourcecostlimitDict= {s.name: fedBidDict[Objects.elfedDict[s.name]][1] for s in Objects.sources}
			edgePriceDict = {e: fedBidDict[Objects.elfedDict[e[1]]][0] for e in nettopObj.edges}
			tempObj = findsolution(nettopObj, tempObj, fedBidDict)
			pathedgelist_fed = findbundles(nettopObj, tempObj)
			solutionObj.federatedPathlist = [[e[0] for e in l] for l in pathedgelist_fed]
			fedValDict = tempObj.fedValDict.copy()
			solutionObj.federatedValues = [v for f,v in sorted(fedValDict.items())]
			# tempObj.reset()
			# testpathlist(pathedgelist_fed, solutionObj_basic)
			# running SLSQP opitmization 
			slsqppricelist_fed, slsqpvaluelist_fed = optimizeSLSQP(nettopObj, pathedgelist_fed, fedValDict, alpha = 0)
			slsqppricelist_cent, slsqpvaluelist_cent = optimizeSLSQP(nettopObj, pathedgelist_cent, fedValDict, alpha = 0)
			
			# assert len(slsqppricelist_fed) == nf and len(slsqppricelist_cent) == nf
			solutionObj.federatedSLSQPValues = slsqpvaluelist_fed
			solutionObj.centralizedSLSQPValues = slsqpvaluelist_cent
			solutionObj.federatedSLSQPPrices = slsqppricelist_fed
			solutionObj.centralizedSLSQPPrices = slsqppricelist_cent
			
			# print("length of pathedgelist_fed:", len(pathedgelist_fed))
			# print("fedBidDict:", fedBidDict)
			fedPriceLowerDict, fedPriceUpperDict = findPriceBoundary(nettopObj, pathedgelist_fed, fedBidDict)
			# binary search for price 
			milpObj, milpPrices = searchPriceBinary(nettopObj, fedPriceLowerDict, fedPriceUpperDict, fedBidDict)
			solutionObj.federatedMILPValues =  [v for f, v in sorted(milpObj.fedValDict.items())]
			solutionObj.federatedMILPPrices = [v for f, v in sorted(milpPrices.items())]
			# for h, so in milpObjDict.items(): 
			# print(Objects.centPriceDict)
			fedPriceLowerDict, fedPriceUpperDict = findPriceBoundary(nettopObj, pathedgelist_cent, Objects.centPriceDict)
			# binary search for price 
			milpObj, milpPrices = searchPriceBinary(nettopObj, fedPriceLowerDict, fedPriceUpperDict, Objects.centPriceDict)
			solutionObj.centralizedMILPValues =  [v for f, v in sorted(milpObj.fedValDict.items())]
			solutionObj.centralizedMILPPrices = [v for f, v in sorted(milpPrices.items())]
			# for p in pathlist: 
			# print("centralized:", solutionObj.centrlaizedValues, sum(solutionObj.centrlaizedValues))
			# print("independent:", solutionObj.independentValues, sum(solutionObj.independentValues))
			# print("federated values:", solutionObj.federatedValues, sum(solutionObj.federatedValues))
			# print("federated MILP:", solutionObj.federatedMILPValues, sum(solutionObj.federatedMILPValues))
			# print("federated MILP prices:", solutionObj.federatedMILPPrices)			
			# print("centralized MILP:", solutionObj.centralizedMILPValues, sum(solutionObj.centralizedMILPValues))
			# print("centralized MILP prices:", solutionObj.centralizedMILPPrices)
			# print("federated SLSQP:", solutionObj.federatedSLSQPValues, sum(solutionObj.federatedSLSQPValues))
			# print("federated SLSQP prices:", solutionObj.federatedSLSQPPrices)
			# print("centralized SLSQP:", solutionObj.centralizedSLSQPValues, sum(solutionObj.centralizedSLSQPValues))
			# print("centralized SLSQP prices:", solutionObj.centralizedSLSQPPrices)
			# print()
			
			procSolObjDict[solhashid] = solutionObj
			if k%10 == 0: 
				with open(proc_sol_filename, 'wb') as outfile: 
					pickle.dump(procSolObjDict, outfile)
				with open(proc_milp_filename, 'wb') as outfile: 
					pickle.dump(milpObjDict, outfile)
		# except:
		# 	continue
			
def multiprocrunAuction(nproc): 
	global sol_filename, milpObjDict, solutionObjDict, milp_filename
	
	filenamelist = (['hashNetworkDict_elements%d_federates%d_density%d'%(numelements, numfederates, edgedivider) 
										for (numfederates, numelements), edgedivider in list(fedeldensitylist)])
	
	proc_filenameDict = {proc: sol_filename.replace('.p','')  + '_proc%s.p'%str(proc).zfill(3) for proc in range(nproc)}
	proc_milpfnameDict = {proc: milp_filename.replace('.p','')  + '_proc%s.p'%str(proc).zfill(3) for proc in range(nproc)}
	# random.shuffle(filenamelist)
	for filename in filenamelist:
		print(filename)
		filename = dir_topologies + filename
		fsolution = filename + '_solutionDict.p'
		ftopnetworks = filename + '_top10.p'
		
		with open(ftopnetworks, 'rb') as infile: 
			nettopObjList = pickle.load(infile)
		
		for nettopObj in nettopObjList:
			try:
				Objects = buildObjects(nettopObj)
				nf = Objects.numfederates
				centralizedObj = LiteSolution(fedBidDict = Objects.centPriceDict, nettopObj = nettopObj)
				centralizedObj = findsolution(nettopObj, centralizedObj, Objects.centPriceDict)
				pathedgelist_cent = findbundles(nettopObj, centralizedObj)
				
				independentObj = LiteSolution(fedBidDict = Objects.indPriceDict, nettopObj = nettopObj)

				independentObj = findsolution(nettopObj, independentObj, Objects.indPriceDict)
				pathedgelist_ind = findbundles(nettopObj, independentObj)
				
				baseObj = {}
				baseObj['centrlaizedValues'] = [v for f, v in sorted(centralizedObj.fedValDict.items())]
				baseObj['independentValues'] = [v for f, v in sorted(independentObj.fedValDict.items())]			
				baseObj['centralizedPathlist'] = [[e[0] for e in l] for l in pathedgelist_cent]
				baseObj['independentPathlist'] = [[e[0] for e in l] for l in pathedgelist_ind]
				baseObj['pathedgelist_cent'] = pathedgelist_cent
				
				allBidList = list(createBid(nf))
				
				N = len(allBidList)
				
				allProcs = []
				for proc in range(nproc):
					inds = range(proc, N, nproc)
					objlist = [allBidList[i] for i in inds]
					p = Process(target=runAuctionProc, args=(baseObj, nettopObj, objlist, proc))
					p.start()
					allProcs.append(p)
				
				for a in allProcs: 
					a.join()
				
				for proc in range(nproc): 
					proc_sol_filename = proc_filenameDict[proc]
					proc_milp_filename = proc_milpfnameDict[proc]
					with open(proc_sol_filename, 'rb') as infile:
					   tempDict = pickle.load(infile)
					   for h, obj in tempDict.items():
					   	solutionObjDict[h] = obj 
					
					with open(proc_milp_filename, 'rb') as infile:
					   tempDict = pickle.load(infile)
					   for h, obj in tempDict.items():
					   	milpObjDict[h] = {**obj, **milpObjDict[h]}

				with open(sol_filename, 'wb') as outfile: 
					pickle.dump(solutionObjDict, outfile)
				
				with open(milp_filename, 'wb') as outfile: 
					pickle.dump(milpObjDict, outfile)
					
			except:
				continue
				
	for proc in range(nproc): 
		proc_sol_filename = proc_filenameDict[proc]
		proc_milp_filename = proc_milpfnameDict[proc]
		with open(proc_sol_filename, 'rb') as infile:
		   tempDict = pickle.load(infile)
		   for h, obj in tempDict.items():
		   	solutionObjDict[h] = obj 
		
		with open(proc_milp_filename, 'rb') as infile:
		   tempDict = pickle.load(infile)
		   for h, obj in tempDict.items():
		   	milpObjDict[h] = {**ob, **milpObjDict[h]}

	with open(sol_filename, 'wb') as outfile: 
		pickle.dump(solutionObjDict, outfile)
	
	with open(milp_filename, 'wb') as outfile: 
		pickle.dump(milpObjDict, outfile)
			
	for proc in range(nproc):
		proc_sol_filename = proc_filenameDict[proc]
		proc_milp_filename = proc_milpfnameDict[proc]
		os.remove(proc_sol_filename)
		os.remove(proc_milp_filename)

def testFiles(): 
	filename = "milpObjDict.p"
	with open(dir_simulations + filename, 'rb') as infile: 
		tempDict = pickle.load(infile)
		# print(tempDict.keys())
		for h, obj in tempDict.items(): 
			print(h)
			for h2, obj2 in obj.items():
				print(h, h2)
				print(obj2.fedValDict)
				print(obj2.fedBidDict)
		
	filename = "solutionObjDict.p"
	with open(dir_simulations + filename, 'rb') as infile: 
		tempDict = pickle.load(infile)
		# print(tempDict.keys())
		for h, obj in tempDict.items(): 
			print(h)
			print(obj.federatedMILPValues)
			print(obj.federatedSLSQPPrices)
	
if __name__ == '__main__':	
	parser = argparse.ArgumentParser(description="This processed raw data of twitter.")
	parser.add_argument('--nproc', type=int, default=3, help='cores on server')
	parser.add_argument('--n', type=int, default=1, help='cores on server')
	args = parser.parse_args()
	argsdict = vars(args)
	nproc = argsdict['nproc']
	
	sol_filename = dir_simulations + 'solutionObjDict.p'
	milp_filename = dir_simulations + 'milpObjDict.p'
	if os.path.isfile(sol_filename): 
	    with open(sol_filename, 'rb') as infile:
	        solutionObjDict = pickle.load(infile)
	else: 
		solutionObjDict = {}
	
	if os.path.isfile(milp_filename): 
	    with open(milp_filename, 'rb') as infile:
	        milpObjDict = pickle.load(infile)
	else: 
		milpObjDict = defaultdict(dict)
		
	Objects = namedtuple('Objects', [])
	multiprocrunAuction(nproc)
	
	# testFiles()
		
	

	