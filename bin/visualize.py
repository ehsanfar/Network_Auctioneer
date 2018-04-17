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
from resources.globalv import * 
from resources.classes import *
import os
import hashlib
from multiprocessing import Process, Manager
import argparse
from collections import defaultdict

def convertSolutionDicts(): 
	global dir_simul, sol_filename, sol_filename_new
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

	
if __name__ == '__main__':	
	parser = argparse.ArgumentParser(description="This processed raw data of twitter.")
	parser.add_argument('--nproc', type=int, default=3, help='cores on server')
	parser.add_argument('--n', type=int, default=1, help='cores on server')
	args = parser.parse_args()
	argsdict = vars(args)
	nproc = argsdict['nproc']
	
	dir_simul = 'simulations/'
	sol_filename = dir_simul + 'solutionObjDict.p'
	sol_filename_new = dir_simul + 'solutionObjDict_new.p'
	convertSolutionDicts()
	
	
	if os.path.isfile(sol_filename_new): 
	    with open(sol_filename_new, 'rb') as infile:
	        solutionObjDict = pickle.load(infile)
	else: 
		solutionObjDict = {}
		 	
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

		print(len(hashlist), len(set(hashlist))) 
		print(k, len(dic))
		# print(len(list(dic.keys())))
		for k2, obj in list(dic.items()):
			# print(len(obj.fedBidDict)) 
			hashlist2.append(k2)
			# print(k2, obj.fedBidDict)#, obj.centralizedValues, obj.independentValues, sum(obj.federatedValues), 
				# sum(obj.centralizedMILPValues), obj.centralizedMILPPrices, obj.centralizedSLSQPPrices, obj.federatedMILPPrices, obj.federatedSLSQPPrices, 
					# obj.federatedPathlist)
		
		print(len(hashlist), len(hashlist2))