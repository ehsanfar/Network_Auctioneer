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
import pickle 
import sys, os 
sys.path.append(os.path.abspath('..'))

from resources.globalv import * 
from resources.classes import *
from resources.generalFunctions import *
import os
import hashlib
from multiprocessing import Process, Manager
import argparse
from collections import defaultdict
import numpy as np 
import matplotlib.pylab as plt 


def convertSolutionDicts(): 
	global dir_simul
	sol_filename = dir_simul + 'solutionObjDict.p'
	sol_filename_new = dir_simul + 'solutionObjDictII.p'
	new_solutionObjDict = defaultdict(dict)

	if os.path.isfile(sol_filename): 
	    with open(sol_filename, 'rb') as infile:
	        solutionObjDict = pickle.load(infile)
	    
	    for k, obj in solutionObjDict.items(): 
	    	if int(k[32:]) in new_solutionObjDict[k[:32]]:
	    		print("repetitive")
	    	new_solutionObjDict[k[:32]][int(k[32:])] = obj
	else: 
		solutionObjDict = {}
	
	# print(len(solutionObjDict))
	with open(sol_filename_new, 'wb') as outfile:
		pickle.dump(new_solutionObjDict, outfile)	


def plotAvgPricevsBid(): 
	global dir_topol
	sol_filename_new = dir_simul + 'solutionObjDictII.p'
	if os.path.isfile(sol_filename_new): 
	    with open(sol_filename_new, 'rb') as infile:
	        solutionObjDict = pickle.load(infile)
	else: 
		solutionObjDict = {}
		
	filenamelist = (['hashNetworkDict_elements%d_federates%d_density%d_top10.p'%(numelements, numfederates, edgedivider) 
										for (numfederates, numelements), edgedivider in list(fedeldensitylist)])
	# print(len(list(solutionObjDict.keys())))
	
	topologies = []
	for filename in filenamelist: 
		with open(dir_topol + filename, 'rb') as infile: 
			topologies.extend(pickle.load(infile))
	
	print("number of topologies:", len(topologies))
	
			
	for k, dic in list(solutionObjDict.items()):
		networkObj = next(x for x in topologies if x.hashid == k)
		# print(networkObj)
		elfedDict = {e: f for e,f in zip(networkObj.elements, networkObj.federates)}
		if len(dic) == 210: 
			selected = []
			avgsellbid = []
			avgbuybid = []
			avgcentvalue = []
			avgfedvalue = []
			avgmilpprice = []
			avgslsprice = []
			
			selected = list(dic.values())
			bidslist = [obj.fedBidDict.values() for obj in selected]
			# print(bidslist)
			avgsellbid = [np.mean([b[0] for b in bids]) for bids in bidslist]
			avgbuybid = [np.mean([b[1] for b in bids]) for bids in bidslist]
			avgfedslsprice = []
			avgfedmilprice = []
			avgcentslsprice = []
			avgcentmilprice = []
			# avgfedslsprice = [np.mean(obj.federatedSLSQPPrices) for obj in selected]
			# avgfedmilprice = [np.mean(obj.federatedMILPPrices) for obj in selected]
			# avgcentslsprice = [np.mean(obj.centralizedSLSQPPrices) for obj in selected]
			# avgcentmilprice = [np.mean(obj.centralizedMILPPrices) for obj in selected]
			avgfedvalues = [np.mean(obj.federatedValues) for obj in selected]
			avgcentvalues = [np.mean(obj.centralizedValues) for obj in selected]
			
			for obj in selected:
				print(obj.hashid)
				fedMilpPriceDict = {'f%d'%i: p for i,p in enumerate(obj.federatedMILPPrices)}
				centMilpPriceDict =  {'f%d'%i: p for i,p in enumerate(obj.centralizedMILPPrices)}
				fedSLSPriceDict =  {'f%d'%i: p for i,p in enumerate(obj.federatedSLSQPPrices)}
				centSLSPriceDict =  {'f%d'%i: p for i,p in enumerate(obj.centralizedSLSQPPrices)} 
				
				print(obj.federatedPathlist)
				print(obj.centralizedPathlist)
				avgfedmilprice.append(calAvgPrice(obj.federatedPathlist, elfedDict, fedMilpPriceDict))
				avgfedslsprice.append(calAvgPrice(obj.federatedPathlist, elfedDict, fedSLSPriceDict))
				avgcentmilprice.append(calAvgPrice(obj.centralizedPathlist, elfedDict, centMilpPriceDict))
				avgcentmilprice.append(calAvgPrice(obj.centralizedPathlist, elfedDict, centSLSPriceDict))
					
			
			
			nethashid = k
			
			sellbuymilp = list(sorted(zip(avgsellbid, avgbuybid, avgfedmilprice, avgfedslsprice))) 
			buyset = sorted(list(set([e[2] for e in sellbuymilp])))
			fig = plt.figure(figsize=(8, 4), dpi=my_dpi)
			ax = fig.add_axes(axes_list_4[j])
			
			for bpr in buyset: 
				plt.plot([e[0] for e in sellbuymilp])
			
			print(nethashid)
			print(sellbuymilp)
	# for obj in selected: 
	# 	print(obj.fedBidDict)
	# 	print(obj.federatedValues)
	# 	print(obj.federatedMILPPrices)
	# 	print(obj.federatedSLSQPPrices)
	# 	print(obj.centralizedMILPPrices)
	# 	print(obj.centralizedSLSQPPrices)

def countObjects(): 
	print(len(list(solutionObjDict.keys())))
	for k, dic in list(solutionObjDict.items()):
		results = createBid(2)
		hashlist = []
		hashlist2 = []
		for r in results:
			pricelist = [e for f, t in sorted(r.items()) for e in t]
			hashid = int(''.join([str(e//10).zfill(3) for e in pricelist]))
			# print(pricelist, hashid)
			hashlist.append(hashid)

		# print(len(hashlist), len(set(hashlist))) 
		print(k, len(dic))
		# print(len(list(dic.keys())))
		# for k2, obj in list(dic.items()):
		# 	print(obj.fedBidDict) 
		# 	hashlist2.append(k2)
			# print(k2, obj.fedBidDict)#, obj.centralizedValues, obj.independentValues, sum(obj.federatedValues), 
				# sum(obj.centralizedMILPValues), obj.centralizedMILPPrices, obj.centralizedSLSQPPrices, obj.federatedMILPPrices, obj.federatedSLSQPPrices, 
					# obj.federatedPathlist)
		# print(len(hashlist), len(hashlist2)

if __name__ == '__main__':	
	parser = argparse.ArgumentParser(description="This processed raw data of twitter.")
	parser.add_argument('--nproc', type=int, default=3, help='cores on server')
	parser.add_argument('--n', type=int, default=1, help='cores on server')
	args = parser.parse_args()
	argsdict = vars(args)
	nproc = argsdict['nproc']
	
	dir_simul = os.path.abspath('..') + '/simulations/'
	dir_topol = os.path.abspath('..') + '/topologies_new/'
	convertSolutionDicts()
	# sol_filename_new = dir_simul + 'solutionObjDictII.p'
	
	# convertSolutionDicts()
	# countObjects()
	plotAvgPricevsBid()
		 	
