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
	sol_filename = dir_simul + 'solutionObjDictIII.p'
	sol_filename_new = dir_simul + 'solutionObjDictIII_new.p'
	new_solutionObjDict = defaultdict(dict)

	if os.path.isfile(sol_filename): 
	    with open(sol_filename, 'rb') as infile:
	        solutionObjDict = pickle.load(infile)
	    
	    for k, obj in solutionObjDict.items(): 
	    	print(k)
	    	print(list(obj.keys()))
	    	if int(k[32:]) in new_solutionObjDict[k[:32]]:
	    		print("repetitive")
	    	new_solutionObjDict[k[:32]][int(k[32:])] = obj
	else: 
		solutionObjDict = {}
	
	# print(len(solutionObjDict))
	with open(sol_filename_new, 'wb') as outfile:
		pickle.dump(new_solutionObjDict, outfile)	


def savePlotDataSet(): 
	global dir_topol
	sol_filename_new = dir_simul + 'solutionObjDictIII.p'
	result_filename = dir_simul + 'resultDict.p'
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
	
	resultDict = {}
	print("length of solutionObjDict: ", len(solutionObjDict))
	for k, dic in list(solutionObjDict.items()):
		networkObj = next(x for x in topologies if x.hashid == k)
		# print(networkObj)
		elfedDict = {e: f for e,f in zip(networkObj.elements, networkObj.federates)}
		# print(len(dic))
		if len(dic) in [210, 1046]: 
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
			selected = [obj for _, obj in zip(avgsellbid, selected)]
			bidslist = [bid for _, bid in zip(avgsellbid, bidslist)]
			avgsellbid = sorted(avgsellbid)
			avgbuybid = [np.mean([b[1] for b in bids]) for bids in bidslist]
			avgfedslsprice = []
			avgfedmilprice = []
			avgcentslsprice = []
			avgcentmilprice = []
			# avgfedslsprice = [np.mean(obj.federatedSLSQPPrices) for obj in selected]
			# avgfedmilprice = [np.mean(obj.federatedMILPPrices) for obj in selected]
			# avgcentslsprice = [np.mean(obj.centralizedSLSQPPrices) for obj in selected]
			# avgcentmilprice = [np.mean(obj.centralizedMILPPrices) for obj in selected]
			avgfedvalues = [sum(obj.federatedValues) for obj in selected]
			avgcentvalues = [sum(obj.centralizedMILPValues) for obj in selected]
			avgindepvalues = [sum(obj.independentValues) for obj in selected]
			print(avgindepvalues)
			print(k)
			for obj in selected:
				# print(obj.hashid)
				fedMilpPriceDict = {'f%d'%i: p for i,p in enumerate(obj.federatedMILPPrices)}
				centMilpPriceDict =  {'f%d'%i: p for i,p in enumerate(obj.centralizedMILPPrices)}
				fedSLSPriceDict =  {'f%d'%i: p for i,p in enumerate(obj.federatedSLSQPPrices)}
				centSLSPriceDict =  {'f%d'%i: p for i,p in enumerate(obj.centralizedSLSQPPrices)} 
				
				# print(obj.federatedPathlist)
				# print(obj.centralizedPathlist)
				avgfedmilprice.append(calAvgPrice(obj.federatedPathlist, elfedDict, fedMilpPriceDict))
				avgfedslsprice.append(calAvgPrice(obj.federatedPathlist, elfedDict, fedSLSPriceDict))
				avgcentmilprice.append(calAvgPrice(obj.centralizedPathlist, elfedDict, centMilpPriceDict))
				avgcentmilprice.append(calAvgPrice(obj.centralizedPathlist, elfedDict, centSLSPriceDict))
					
			
			nethashid = k
			titles = ['numfederates', 'sell', 'buy', 'fedmilp', 'centmilp', 'fedsls', 'centsls', 'fedvalue', 'centvalue', 'indepvalue']
			if len(dic) == 210:
				numf = 2
			elif len(dic) == 1046: 
				print("length 1046")
				numf = 3
			else: 
				numf == None
				
			resultlist = [numf, avgsellbid, avgbuybid, avgfedmilprice, avgcentmilprice, avgfedslsprice, avgcentslsprice, avgfedvalues, avgcentvalues, avgindepvalues]
			# sellbuymilp = list(zip(list(sorted(zip(avgsellbid, avgbuybid, avgfedmilprice, avgcentmilprice, avgfedslsprice, avgcentslsprice))) 
			resultDict[k] = dict(zip(titles, resultlist))
			# buyset = sorted(list(set([e[2] for e in sellbuymilp])))
			# fig = plt.figure(figsize=(8, 4), dpi=my_dpi)
			# ax = fig.add_axes(axes_list_2[0])
			# for bpr in buyset: 
			# 	plt.plot([e[0] for e in sellbuymilp])
			
	with open(result_filename, 'wb') as outfile:
		pickle.dump(resultDict, outfile)
			
			# print(nethashid)
			# print(sellbuymilp)
	# for obj in selected: 
	# 	print(obj.fedBidDict)
	# 	print(obj.federatedValues)
	# 	print(obj.federatedMILPPrices)
	# 	print(obj.federatedSLSQPPrices)
	# 	print(obj.centralizedMILPPrices)
	# 	print(obj.centralizedSLSQPPrices)

def countObjects(): 
	# print(len(list(solutionObjDict.keys())))
	sol_filename_new = dir_simul + 'solutionObjDictIII.p'
	if os.path.isfile(sol_filename_new): 
	    with open(sol_filename_new, 'rb') as infile:
	        solutionObjDict = pickle.load(infile)
	else: 
		solutionObjDict = {}
		
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

def plotResults(): 
	global dir_simul, dir_fig
	result_filename = dir_simul + 'resultDict.p'
	
	titles = ['sell', 'buy', 'fedmilp', 'centmilp', 'fedsls', 'centsls', 'fedvalue', 'centvalue', 'indepvalue']
	
	with open(result_filename, 'rb') as infile:
	    resultDict = pickle.load(infile)
				
	
	aggdict2 = defaultdict(list)
	aggdict3 = defaultdict(list)
	# print(resultDict)
	for k, dic in list(resultDict.items()): 
		# print(dic)
		for title in titles:
			# print(dic['numfederates'])
			# print(dic[title])
			if dic['numfederates'] == 2:
				aggdict2[title].extend(dic[title])
			
			elif dic['numfederates'] == 3:
				aggdict3[title].extend(dic[title])
	
	aggdiclist = [aggdict2, aggdict3]
	
		
	for i, dic in zip([2,3], aggdiclist): 
		print([len(e) for e in dic.values()])
		fig = plt.figure(figsize=(8, 4), dpi=my_dpi)
		buyset = set(dic['buy'])
		print(len(dic['sell']), len(dic['fedmilp']), len(dic['buy']))
		baselist = list(range(500, 1001, 100))
		buyranges = list(zip(baselist, baselist[1:]))
		plots = ['fedmilp', 'fedsls']
		plottitles = ['milp', 'slsqp']
		for j, pl in enumerate(plots): 
			ax = fig.add_axes(axes_list_2[j])
			for brange in buyranges:
				print(brange)
				sell, fedmilp = zip(*[(sell, fedmilp) for sell, fedmilp, b in zip(dic['sell'], dic[pl], dic['buy']) if brange[0]<=b<brange[1] and 100<=sell<=500])
				# print(len(sell), len(fedmilp)) 
				# print(sell)
				sell, avgfedmilp = groupbylists(sell, fedmilp, func = 'avg')#sorted([(sell, np.mean(avgmilp)) for sell, avgmilp in groupby(zip(dic['sell'], dic['fedmilp']), itemgetter(0))])
				print(sell)
				print(avgfedmilp)
				# print(sell)
				# for buy in buyset:
				plt.plot(sell, avgfedmilp)
			
			if i == 2:
				plt.ylim(320, 520)
			if i == 3: 
				plt.ylim(350, 470)			
			plt.xlim(100, 500)
			plt.plot([0, 500], [0, 500], '--')
			plt.title(plottitles[j])
			if j == 1:
				plt.legend(['Path bid : %s-%s'%(str(tup[0]), str(tup[1])) for tup in buyranges] + ['Avg bid line'])
			else: 
				plt.ylabel('average actualized link price')
				plt.xlabel('average federated link bids')
		# ax = fig.add_axes(axes_list_2[1]) 
		# plt.scatter(dic['sell'], dic['fedsls'])
		plt.savefig(dir_fig + 'sell_fedprice_numfeds%s.png'%str(i).zfill(2), format='png', dpi=my_dpi, bbox_inches='tight')
	
	plottitles = ['fedvalue', 'centvalue']
	fig = plt.figure(figsize=(8, 4), dpi=my_dpi)

	for i, dic in zip([2,3], aggdiclist): 
		# print([len(e) for e in dic.values()])
		buyset = set(dic['buy'])
		# print(len(dic['sell']), len(dic['fedmilp']), len(dic['buy']))
		baselist = list(range(500, 1001, 100))
		buyranges = list(zip(baselist, baselist[1:]))
		ax = fig.add_axes(axes_list_2[i-2])
		for brange in buyranges:
			# print(brange)
			sell, fedmilp = zip(*[(sell, fedmilp) for sell, fedmilp, b in zip(dic['sell'], dic['fedvalue'], dic['buy']) if brange[0]<=b<brange[1] and 100<=sell<=500])
			# print(len(sell), len(fedmilp)) 
			# print(sell)
			sell, avgfedmilp = groupbylists(sell, fedmilp, func = 'avg')#sorted([(sell, np.mean(avgmilp)) for sell, avgmilp in groupby(zip(dic['sell'], dic['fedmilp']), itemgetter(0))])
			# print(sell)
			# print(avgfedmilp)
			# print(sell)
			# for buy in buyset:
			plt.plot(sell, avgfedmilp)
			# plt.ylim(300, 550)
		
		# plt.plot([0, 500], [0, 500], '--')
		plt.title('number of federates: %d'%i)
		
		avgcentvalue = np.mean(dic['centvalue'])
		avgindepvalue = np.mean(dic['indepvalue'])
		# plt.ylim(7000, )
		plt.axhline(y=avgcentvalue, linestyle = '--')
		plt.axhline(y=avgindepvalue, linestyle = '-.')
		
		if i == 3:
			plt.legend(['Path bid : %s-%s'%(str(tup[0]), str(tup[1])) for tup in buyranges] + ['centralized', 'independent'])
		else: 
			plt.ylabel('avg value')
			plt.xlabel('avg link bid')
		# ax = fig.add_axes(axes_list_2[1]) 
		# plt.scatter(dic['sell'], dic['fedsls'])
	plt.savefig(dir_fig + 'sell_fedcentvalues_numfeds.png', format='png', dpi=my_dpi, bbox_inches='tight')
	
	
		
if __name__ == '__main__':	
	parser = argparse.ArgumentParser(description="This processed raw data of twitter.")
	parser.add_argument('--nproc', type=int, default=3, help='cores on server')
	parser.add_argument('--n', type=int, default=1, help='cores on server')
	args = parser.parse_args()
	argsdict = vars(args)
	nproc = argsdict['nproc']
	
	dir_fig = os.path.abspath('..') + '/figures/'
	dir_simul = os.path.abspath('..') + '/simulations/'
	dir_topol = os.path.abspath('..') + '/topologies_new/'
	# convertSolutionDicts()
	# sol_filename_new = dir_simul + 'solutionObjDictII.p'
	
	# convertSolutionDicts()
	# countObjects()
	# plotAvgPricevsBid()
	# savePlotDataSet()
	plotResults()
		 	
# 