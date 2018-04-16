from resources.classes import *
from resources.globalv import * 
from collections import defaultdict, Counter
from itertools import product 
import pickle
import os
import random
import hashlib
from resources.optimizeMILP import optimizeMILP
from multiprocessing import Process, Manager
import argparse
import os

dir_topologies = 'topologies_new/' 

def createNetTopologies(): 	
	global 	seedlist, filename, numfederates, elementnames, edgedivider
	numberfederates = numfederates*[len(elementnames)//numfederates]			# print([s.name for s in sources])
	destinations = elementnames[-2:]	
	sources = [e for e in elementnames if e not in destinations]
	
	if os.path.isfile(filename): 
	    with open(filename, 'rb') as infile:
	        hashNetworkDict = pickle.load(infile)
	    
	    hashNetworkDict = {h: obj for h,obj in hashNetworkDict.items() if obj.costValueDict}
	    	
	else: 
		hashNetworkDict = {}
		
	for seed in seedlist:
		# print(seed)
		random.seed(seed)
		while sum(numberfederates)<len(elementnames): 
			i = random.choice(range(len(numberfederates)))
			numberfederates[i] += 1
		
		namelist = [n*['f%d'%i] for i, n in enumerate(numberfederates)]
		federatenames = [e for l in namelist for e in l]
		random.shuffle(federatenames) 
		# print("shuffle:", federatenames)
		# all_edges = [(satellites[0],satellites[1]), (satellites[3],stations[0]), (satellites[1],satellites[3]),
		#              (satellites[2],satellites[4]), (satellites[2],satellites[1]), (satellites[2],satellites[3]), (satellites[3],satellites[4]), (satellites[4],stations[1]), (satellites[2],stations[0])]
		# all_possible_edges = [(a,b) for a, b in list(product(elementnames, elementnames)) if (a != b and element_federate_dict[a] != element_federate_dict[b])]
		all_possible_edges = []
		all_edges = []
		# while len([l for l in all_possible_edges if l[1] in destinations])<len(elements)//linkcapacity:
		all_possible_edges = [(a,b) for a, b in list(product(elementnames, elementnames)) if (a != b and not (a in destinations))]
		all_edges = random.sample(all_possible_edges, int(len(all_possible_edges)//edgedivider))
		edge2destin = [l for l in all_possible_edges if l[1] in destinations and l not in all_edges]
		existingedges2desgin = [l for l in all_edges if l[1] in destinations]
		nume2d = int(len(sources)/2 - len(existingedges2desgin))
		# print(nume2d)
		if nume2d>0:
			newedges = random.sample(edge2destin, nume2d)
			# print(len(all_edges))
			all_edges = all_edges + newedges			
			# print(newedges)
			# print(len(all_edges))
		
		all_edge_set = set([])
		destin_count = 0

		for edge in all_edges:
		    s, d = edge
		    # if destin_count > len(satellites):
		    #     continue
		    if s in destinations or d in destinations:
		        destin_count += linkcapacity
		    all_edge_set.add((s,d))
		    all_edge_set.add((d,s))

		all_edges = list(all_edge_set)
		
		tempNetTop = NetTop(elementnames, all_edges, federatenames, sources, destinations)
		if tempNetTop.hashid not in hashNetworkDict: 
			# print(seed, tempNetTop.hashid)
			hashNetworkDict[tempNetTop.hashid] = tempNetTop
	
	with open(filename, 'wb') as outfile: 
		pickle.dump(hashNetworkDict, outfile)


def calCostValue(nettopObj): 
	federatenames = nettopObj.federates
	# fedPriceDict = {fname: (sharelinkcost, uselinkcost) for fname in federatenames}
	# federates = [Federate(name = f, cash = 0, sharelinkcost = fedPriceDict[f][0], uselinkcost = fedPriceDict[f][1]) for f in set(federatenames)]
	# federateDict = {f.name: f for f in federates}
	# # print("element names:", nettopObj.elements)
	
	# elements = [Element(name = e, capacity=elementcapacity, size = 0, owner = federateDict[f]) for (e,f) in zip(nettopObj.elements, federatenames)]
	# elementDict = {e.name: e for e in elements}
	# sources = [e for e in elements if e.name not in nettopObj.destinations]
	# # sources = nettopObj.sources
	# # print([s.name for s in sources])
	# linklist = [Link(source = elementDict[e1], destin = elementDict[e2], capacity = linkcapacity, size = 0, owner = elementDict[e2].owner) for (e1, e2) in nettopObj.edges]
	# time = 0
	# newtasks = [Task(id = id + n, element=s, lastelement=s, size=size, value=value, expiration=time + 5, init=time, active=True, penalty=penalty) for n, s in enumerate(sources)]
	# elfedDict = {e: f for e, f in zip(nettopObj.elements, nettopObj.federates)}


	# federates = [Federate(name = f, cash = 0, sharelinkcost = 0, uselinkcost = 0) for f in set(federatenames)]
	# federateDict = {f.name: f for f in federates}
	# # print("element names:", nettopObj.elements)
	# elements = [Element(name = e, capacity=elementcapacity, size = 0, owner = federateDict[f]) for (e,f) in zip(nettopObj.elements, federatenames)]
	# elementDict = {e.name: e for e in elements}
	# sources = [e for e in elements if e.name not in nettopObj.destinations]
	# # sources = nettopObj.sources
	# # print([s.name for s in sources])
	# linklist = [Link(source = elementDict[e1], destin = elementDict[e2], capacity = linkcapacity, size = 0, owner = elementDict[e2].owner) for (e1, e2) in nettopObj.edges]
	# time = 0
	# newtasks = [Task(id = id + n, element=s, lastelement=s, size=size, value=value, expiration=time + 5, init=time, active=True, penalty=penalty) for n, s in enumerate(sources)]
	
	# elfedDict = {e: f for e, f in zip(nettopObj.elements, nettopObj.federates)}
	# print(elfedDict)
	# print("new tasks:", newtasks)
	for sharelinkcost, uselinkcost in basetuples:
		fedPriceDict = {fname: (sharelinkcost, uselinkcost) for fname in federatenames}

		federates = [Federate(name = f, cash = 0, sharelinkcost = fedPriceDict[f][0], uselinkcost = fedPriceDict[f][1]) for f in set(federatenames)]
		federateDict = {f.name: f for f in federates}
		# print("element names:", nettopObj.elements)
		
		elements = [Element(name = e, capacity=elementcapacity, size = 0, owner = federateDict[f]) for (e,f) in zip(nettopObj.elements, federatenames)]
		elementDict = {e.name: e for e in elements}
		sources = [e for e in elements if e.name not in nettopObj.destinations]
		# sources = nettopObj.sources
		# print([s.name for s in sources])
		linklist = [Link(source = elementDict[e1], destin = elementDict[e2], capacity = linkcapacity, size = 0, owner = elementDict[e2].owner) for (e1, e2) in nettopObj.edges]
		time = 0
		newtasks = [Task(id = id + n, element=s, lastelement=s, size=size, value=value, expiration=time + 5, init=time, active=True, penalty=penalty) for n, s in enumerate(sources)]
		elfedDict = {e: f for e, f in zip(nettopObj.elements, nettopObj.federates)}
		# print("new tuple:", sharelinkcost, uselinkcost)
		# print("length of cost value dict:", len(nettopObj.costValueDict))
		# print(nettopObj.hashid, nettopObj.costValueDict)
		# if (sharelinkcost, uselinkcost) not in nettopObj.costValueDict or nettopObj.costValueDict[(sharelinkcost, uselinkcost)] == 0:
			# for f in federates: 
			# 	f.cash = 0 
			# 	f.sharelinkcost = sharelinkcost
			# 	f.uselinkcost = uselinkcost
		edgePriceDict = {e: fedPriceDict[elfedDict[e[1]]][0] for e in nettopObj.edges}
		# print(edgePriceDict)
		# print(nettopObj.hashid)
		# print(fedPriceDict)
		# print(linklist)
		# print(nettopObj.destinations)
		# print(len(newtasks))
		# print(federates)
		# print(linklist)
		solutionObj = MILPSolution(nettopObj.hashid, time, fedPriceDict = fedPriceDict, fedValDict = {f: 0 for f in fedPriceDict.keys()}, edgelist = [])
		
		solutionObj = optimizeMILP(elements = elements, linklist = linklist, destinations = nettopObj.destinations, 
		storedtasks = [], newtasks = newtasks, time = time, federates = federates, edgePriceDict = edgePriceDict, 
		solutionObj = solutionObj)
		
		totalvalue = solutionObj.totalvalue
		# print(solutionObj.sourceEdgeDict)
		# print(solutionObj.fedValDict)
		nettopObj.costValueDict[(sharelinkcost, uselinkcost)] = totalvalue
			# print("New tuple cost and value:", sharelinkcost, uselinkcost, totalvalue)
	

def updateCostValue(objlist, proc, tempfilename): 
	global filename
	if os.path.isdir("/home/abbas.ehsanfar/gurobi"):
		hostname = os.environ['HOSTNAME']
		os.environ['GRB_LICENSE_FILE'] = "/home/abbas.ehsanfar/gurobi/%s/lic%s/gurobi.lic"%(hostname,str(proc%30).zfill(2))
		
	for k, nettopObj in enumerate(objlist):
		# print("New topoology:", nettopObj.hashid)
		calCostValue(nettopObj) 
		if k%20 == 0:
			objDict = {obj.hashid: obj for obj in objlist}
			with open(tempfilename, 'wb') as outfile: 
				pickle.dump(objDict, outfile)
			
			# with open(filename, 'rb') as infile: 
			# 	hashNetworkDict = pickle.load(infile)
							
			# for h, obj in objDict.items():
			# 	hashNetworkDict[h] = obj
				
			# with open(filename, 'wb') as outfile: 
			# 	pickle.dump(hashNetworkDict, outfile)
			
	with open(tempfilename, 'wb') as outfile: 
		objDict = {obj.hashid: obj for obj in objlist}
		pickle.dump(objDict, outfile)


def multiProcCostValue(): 
	global nproc, filename
	with open(filename, 'rb') as infile: 
		hashNetworkDict = pickle.load(infile)
	
	topollist = list(hashNetworkDict.values())
	N = len(topollist)
	
	allProcs = []
	for proc in range(nproc):
		tempfilename = filename[:-2] + '_proc%s.p'%str(proc).zfill(2)
		inds = range(proc, N, nproc)
		objlist = [topollist[i] for i in inds]
		p = Process(target=updateCostValue, args=(objlist,proc,tempfilename))
		p.start()
		allProcs.append(p)
	
	for a in allProcs: 
		a.join()
		
	finalDict = {}
	for proc in range(nproc): 
		tempfilename = filename[:-2] + '_proc%s.p'%str(proc).zfill(2)
		with open(tempfilename, 'rb') as infile:
		    hashNetworkDict = pickle.load(infile)
		
		for h, obj in hashNetworkDict.items(): 
			finalDict[h] = obj 
			   
	with open(filename, 'wb') as outfile: 
		pickle.dump(finalDict, outfile)
	
	for proc in range(nproc):
		os.remove(filename[:-2] + '_proc%s.p'%str(proc).zfill(2))

def calAuctionScore(): 
	global filename
	print(filename)
	if os.path.isfile(filename): 
		with open(filename, 'rb') as infile:
			hashNetworkDict = pickle.load(infile)
		  # topollist = hashNetworkDict.values()
		# if len(hashNetworkDict) < 1000:
		# 	return 
		   
		for k, topol in hashNetworkDict.items(): 
			costdict = topol.costValueDict
			
			maxtup = (0, 1000)
			for mintup in [(500,501), (400,600)]: 
				if mintup in costdict: 
					topol.auctionscore += costdict[maxtup] - costdict[mintup]
			
			# print(topol.auctionscore)

		toplist = sorted(hashNetworkDict.values(), key = lambda x: x.auctionscore, reverse = True)[:10]
		# print(filename, [e.auctionscore for e in toplist])
		with open(filename[:-2] + '_top10.p', 'wb') as outfile: 
			pickle.dump(toplist, outfile)


		with open(filename[:-2] + '_score.p', 'wb') as outfile: 
			pickle.dump(hashNetworkDict, outfile)
	    				
	else:
		return 	

def aggregate60Nodes(): 
	for numfederates, numelements, edgedivider in [(4,20,7), (4,20,11), (4,20,3), (4,20,5)]: 
		filename = dir_topologies + 'hashNetworkDict_elements%d_federates%d_density%d.p'%(numelements, numfederates, edgedivider)
		finalDict = {}
		for proc in range(60): 
			tempfilename = filename[:-2] + '_proc%s.p'%str(proc).zfill(2)
			with open(tempfilename, 'rb') as infile:
				hashNetworkDict = pickle.load(infile)
			
			for h, obj in list(hashNetworkDict.items()): 
				finalDict[h] = obj 
		
		with open(filename, 'wb') as outfile: 
			pickle.dump(finalDict, outfile)
			
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="This processed raw data of twitter.")
	parser.add_argument('--nproc', type=int, default=3, help='cores on server')
	parser.add_argument('--n', type=int, default=3, help='cores on server')
	args = parser.parse_args()
	argsdict = vars(args)
	nproc = argsdict['nproc']
	
	time = 0
	# basecost = [0, 200, 400, 600, 800, 1000]
	seedlist = list(range(0,500))
	# for (numfederates, numelements), edgedivider in reversed(list(product([(2,10), (2,15), (3,15), (2,20), (3,20), (4,20)], [3,5,7,11]))): 
	for (numfederates, numelements), edgedivider in list(fedeldensitylist): 
		filename = dir_topologies + 'hashNetworkDict_elements%d_federates%d_density%d.p'%(numelements, numfederates, edgedivider)
		elementnames = ['e%d'%(i+1) for i in range(numelements)]
		# createNetTopologies()
		# multiProcCostValue() 
		calAuctionScore()