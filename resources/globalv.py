import numpy as np 
from itertools import product
import random 

dir_figures = 'figures/'

# basecost = [200, 600]
dif = 100
seed = 6
random.seed(seed) 
basecost = [0, 1000]
basetuples = [(0, 1000), (1000, 999), (500, 501), (400, 600), (200, 800)]
centPriceDict = {}
fedeldensitylist = product([(2,10), (2,15), (3,15), (2,20), (3,20), (4,20)], reversed([3,5,7,11]))
basebids = list(product(list(np.arange(dif, 501, dif)), np.arange(500, 1001, dif)))
# print(basebids)
# basebids =  [(0, 1000), (50, ), (500, 501), (400, 600), (200, 800)]
experiment = 'fixed'
# federatenames = ['F1', 'F2']
# numfederates = len(federatenames)
# elementnames = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10']

storagepenalty = 100
epsilon = 10
linkcapacity = 2
elementcapacity = 2
taskvalue = maxvalue = 1000

id = 1
SP = 100
epsilon = 10
linkcost = 1001
storagepenalty = 100
value = 1000
penalty = -200
size = 1

dpri = 50
firstprice = maxvalue
time = 0

def createBid(numf): 
	global basebids
	tempDict = {}
	# random.shuffle(basebids)
	templist = numf * [basebids]
	bidcombinatorics = product(*templist)
	# random.shuffle(bidcombinatorics)
	yield {'f%d'%i: (0, 1000) for i in range(numf)}
	yield {'f%d'%i: (1000, 100) for i in range(numf)}
	for tuptup in bidcombinatorics: 
		minedgecost = min([tup[0] for tup in tuptup])
		maxedgecost = max([tup[0] for tup in tuptup])
		minpathcost = min([tup[1] for tup in tuptup])
		maxpathcost = max([tup[1] for tup in tuptup])
		if maxedgecost - minedgecost > 100 or maxpathcost - minpathcost > 100 or maxedgecost == 0 : 
			continue 
				
		yield {'f%d'%i: tup for i, tup in enumerate(tuptup)}


# print(len(list(createBid(2))))
# print(len(list(createBid(3))))
# print(len(list(createBid(4))))

s_1 = s_3 = 0.05
s_5 = 0.04
s_2 = 0.05
d_2 = 0.5
d_3 = 0.33
d_4 = 0.25
d_5 = 0.2
d_6 = 0.15
w_1 = 0.85
w_2 = 0.4
w_3 = 0.28
w_4 = 0.225
w_5 = 0.16
w_6 = 0.11
axes_list_2 =  [[s_2, s_1, w_2, w_1], [s_2+1*d_2, s_1, w_2, w_1]]
my_dpi = 300
