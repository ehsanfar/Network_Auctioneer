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

from scipy.optimize import minimize
from .classes import *
from .globalv import *
import numpy as np 

def optimizeSLSQP(nettopObj, pathedgelist, fedValDict, alpha = 1): 
	global epsilon
	# fedValDict = solutionObj.fedValDict
	elfedDict = {e: f for e, f in zip(nettopObj.elements, nettopObj.federates)}
	pricelist = len(set(nettopObj.federates)) * [0]#[np.mean(list(solutionObj.fedPriceDict.values()))]
	objective = Objective(pathedgelist, elfedDict, alpha = alpha)
	conslist1 = [{'type': 'ineq', 'fun': Constraint1('f%d'%i, pathedgelist, elfedDict, fedValDict['f%d'%i])} for i in range(len(pricelist))]
	conslist2 = [{'type': 'ineq', 'fun': Constraint2(edgelist, elfedDict)} for i, edgelist in enumerate(pathedgelist)]
	conslist3 = [{'type': 'ineq', 'fun': Constraint3(pathedgelist, elfedDict)}]
	cons = conslist1 + conslist2 + conslist3# [con1, con2, con3][:len(initCostDict)]
	# cons = conslist1 + conslist2 # [con1, con2, con3][:len(initCostDict)]
	bnds = [(epsilon, 1000) for c in pricelist]
	sol = minimize(objective, pricelist, method = 'SLSQP', bounds = bnds, constraints = cons)
	
	fedvaluelist = [v for f,v in sorted(fedValDict.items())]
	slsqpvalulist = [None for f in fedvaluelist]
	try:
		slsqpvalulist = [int(con['fun'](sol.x) + fedvaluelist[i]) for i, con in enumerate(conslist1)]
		# for con in conslist1: 
			# fedValueList.append()

		# cons_changes = []
		# for con in cons:
		#   if not con['fun'](sol.x):
		#   	continue
		#   cons_changes.append(int(round(con['fun'](sol.x))))
		#   # consresults = all([e >= 0 for e in [int(round(con['fun'](sol.x))) for con in cons]])
		# if all([e >= 0 for e in cons_changes]) and sum(cons_changes[:2])>0:
		#   # if True:
	except: 
		print("slsqp didn't work")
	# fedPriceDict = {'f%d'%i: int(price) if price else None for i, price in enumerate(list(sol.x))}
	return ([int(e) for e in list(sol.x)], slsqpvalulist)