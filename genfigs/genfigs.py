import pymongo
import socket
import json
# import pandas as pd
# import scipy.stats as stats
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from collections import defaultdict, deque
import math
import numpy as np
# from matplotlib.font_manager import FontProperties
import networkx as nx
# import matplotlib.animation
from itertools import product, chain
import pickle
import random
from numpy import convolve
plt.rcParams.update({'font.size': 16})
from matplotlib.ticker import FormatStrFormatter
import os, sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.generalFunctions import *
sys.path.append(os.path.abspath('..'))



hardcoded_designs = (
        "1.GroundSta@SUR1 2.GroundSta@SUR4 1.Sat@MEO1 1.Sat@MEO4 2.Sat@MEO5 1.Sat@LEO1 2.Sat@LEO2",
        "1.GroundSta@SUR1 2.GroundSta@SUR4 2.Sat@GEO4 1.Sat@MEO1 1.Sat@MEO4 2.Sat@MEO5 1.Sat@LEO1 2.Sat@LEO2",
        "1.GroundSta@SUR1 2.GroundSta@SUR3 3.GroundSta@SUR5 1.Sat@MEO1 1.Sat@MEO2 2.Sat@MEO3 2.Sat@MEO5 3.Sat@MEO6",
        "1.GroundSta@SUR1 2.GroundSta@SUR3 3.GroundSta@SUR5 3.Sat@GEO5 1.Sat@MEO1 1.Sat@MEO2 2.Sat@MEO3 2.Sat@MEO5 3.Sat@MEO6",
        # "1.GroundSta@SUR1 2.GroundSta@SUR3 3.GroundSta@SUR5 1.Sat@MEO1 2.Sat@MEO2 3.Sat@MEO5 1.Sat@LEO2 2.Sat@LEO4 3.Sat@LEO6",
        # "1.GroundSta@SUR1 2.GroundSta@SUR3 3.GroundSta@SUR5 1.Sat@GEO1 1.Sat@MEO1 2.Sat@MEO4 3.Sat@MEO5 2.Sat@LEO4 3.Sat@LEO6",
        "1.GroundSta@SUR1 2.GroundSta@SUR3 3.GroundSta@SUR5 1.Sat@MEO1 2.Sat@MEO2 3.Sat@MEO3 1.Sat@LEO1 2.Sat@LEO2 3.Sat@LEO3",
    )


design_dict = {d: i for i,d in enumerate(hardcoded_designs)}
xtickDict = {0: 'I', 1: 'II', 2: 'III', 3: 'IV', 4: 'V', 5: 'VI', 6: 'VII', 7: 'VIII', 8: 'XI', 9: 'X'}
xticklist = ['Design %s' % xtickDict[i] for i in range(len(hardcoded_designs))]
# divider = 1000000
romannumdict = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V'}


class DesignClass():
    def __init__(self, id, query):
        self.id = id
        self.query = query
        self.totalcash = []
        self.federatecash = []
        self.biddings = []
        self.prices = []
        self.links = []
        self.avgprices = []
        self.avgbiddings = []
        self.totallinks = []
        self.values = []
        self.totalvalues = []


def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma

def filterInvalidDics(dictlist):
    return [dic for dic in dictlist if len(set(dic['costDictList'])) <= 2]

def findBalancedMembers(dictlist):
    return [dic for dic in dictlist if len(set(dic['costDictList'])) == 1]

def fops2costs(fops):
    costSGL, storagePenalty, auctioneer = re.search('x([-\d]+),([-\.\d]+),([-\d]+)', fops).groups()
    # print(costSGL, costISL, storagePenalty)
    return (int(costSGL), float(storagePenalty), int(auctioneer))

def fopsGen(des, test):
    # print("test:",test)
    numPlayers = 2
    if '3.' in des:
        numPlayers = 3

    if 'regular storage' in test:
        if 'stochastic' in test or 'random' in test:
            costsgl = [-3]
        else:
            costsgl = [600]

        storage = [-1, 400, 800]
        for sgl in costsgl:
            for stor in storage:
                fopslist = numPlayers*['x%d,%1.2f,%d'%(sgl, stor, -1)]
                yield fopslist
    # print("design:", des, numPlayers)

    elif 'storage' in test.lower():
        if 'stochastic' in test.lower():
            yield numPlayers * ["x%d,%1.2f,%d" % (-3, 400, -1)]
            yield numPlayers * ["x%d,%1.2f,%d" % (-3, 800, -1)]
            for k in np.linspace(0., 1.99, 19):
                yield numPlayers * ["x%d,%1.2f,%d" % (-3, -1*k, -1)]
        else:
            yield numPlayers * ["x%d,%1.2f,%d" % (600, 400, -1)]
            yield numPlayers * ["x%d,%1.2f,%d" % (600, 800, -1)]
            for k in np.linspace(0., 1.99, 19):
                yield numPlayers * ["x%d,%1.2f,%d" % (600, -1*k, -1)]

    elif 'federate adaptive' in test:
        costrange = [10, 1200, 600]
        for sgl in costrange:
            fops_adaptive = ['x%d,%d,%d' % (-2, -1, -1)] + (numPlayers-1) * ['x%d,%d,%d' % (sgl, -1, -1)]
            fops = numPlayers * ['x%d,%d,%d' % (sgl, -1, -1)]
            yield (fops, fops_adaptive)

    elif 'total' in test:
        costrange = [0, 600, 1200]
        # print(costrange)
        for sgl in costrange:
            # print(sgl)
            if numPlayers == 2:
                fops_1 = ['x%d,%d,%d' % (sgl, -1, -1), 'x%d,%d,%d' % (sgl, -1, -1)]
                fops_2 = ['x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (sgl, -1, -1)]
                fops_3 = ['x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (-2, -1, -1)]
                print("new")
                yield (fops_1, fops_2, fops_3)
            elif numPlayers == 3:
                fops_1 = ['x%d,%d,%d' % (sgl, -1, -1), 'x%d,%d,%d' % (sgl, -1, -1), 'x%d,%d,%d' % (sgl, -1, -1)]
                fops_2 = ['x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (sgl, -1, -1), 'x%d,%d,%d' % (sgl, -1, -1)]
                fops_3 = ['x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (sgl, -1, -1)]
                fops_4 = ['x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (-2, -1, -1)]
                yield (fops_1, fops_2, fops_3, fops_4)

    elif 'auctioneer' in test:
        fops_0 = numPlayers * ['x%d,%d,%d' % (0, -1, -1)]
        fops_1 = numPlayers * ['x%d,%d,%d' % (-2, -1, -1)]
        fops_2 = numPlayers * ['x%d,%d,%d' % (-2, -1, 1)]
        yield (fops_0, fops_1, fops_2)



def fopsGenTotal(numPlayers, numAdaptive, cost): # NA: number of adaptive
    # numPlayers = 2
    # if '3.' in des:
    #     numPlayers = 3
    #
    # costrange = [0, 600, 1200]
    #
    # for sgl in costrange:
    if numPlayers == 2:
        if numAdaptive == 1:
            yield ['x%d,%d,%d' % (cost, -1, -1), 'x%d,%d,%d' % (-2, -1, -1)]
            yield ['x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (cost, -1, -1)]

        elif numAdaptive == 2:
            yield ['x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (-2, -1, -1)]

        elif numAdaptive == 0:
            yield ['x%d,%d,%d' % (cost, -1, -1), 'x%d,%d,%d' % (cost, -1, -1)]

    elif numPlayers == 3:
        if numAdaptive == 0:
            yield ['x%d,%d,%d' % (cost, -1, -1), 'x%d,%d,%d' % (cost, -1, -1), 'x%d,%d,%d' % (cost, -1, -1)]

        elif numAdaptive == 1:
            yield ['x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (cost, -1, -1), 'x%d,%d,%d' % (cost, -1, -1)]
            yield ['x%d,%d,%d' % (cost, -1, -1), 'x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (cost, -1, -1)]
            yield ['x%d,%d,%d' % (cost, -1, -1), 'x%d,%d,%d' % (cost, -1, -1), 'x%d,%d,%d' % (-2, -1, -1)]

        elif numAdaptive == 2:
            yield ['x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (cost, -1, -1)]
            yield ['x%d,%d,%d' % (cost, -1, -1), 'x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (-2, -1, -1)]
            yield ['x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (cost, -1, -1), 'x%d,%d,%d' % (-2, -1, -1)]

        elif numAdaptive == 3:
            yield ['x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (-2, -1, -1), 'x%d,%d,%d' % (-2, -1, -1)]


def convertLocation2xy(location):
    if 'SUR' in location:
        r = 0.5
    elif 'LEO' in location:
        r = 1.
    elif 'MEO' in location:
        r = 1.5
    elif "GEO" in location:
        r = 2
    else:
        r = 2.35

    sect = int(re.search(r'.+(\d)', location).group(1))
    tetha = +math.pi / 3 - (sect - 1) * math.pi / 3

    x, y = (r * math.cos(tetha), r * math.sin(tetha))
    # print location, x, y
    return (x, y)


def convertRadial2xy(r, tetha):
    x, y = (r * math.cos(tetha), r * math.sin(tetha))
    return (x, y)



def createPoints(numPlayers, numAdaptive, i, y, xdelta, ydelta, k):
    # print(letters)
    letterdict = defaultdict(list)
    x = i+1
    d = 0.35
    k = (1.+k)/2.
    xdelta = k*xdelta
    ydelta = k*ydelta/.12
    points = []
    if numPlayers == 2:
        if numAdaptive == 0:
            delta = -d
        elif numAdaptive == 1:
            delta = 0
        elif numAdaptive == 2:
            delta = d

        xlist = [delta + x-xdelta/2., delta + x + xdelta/2.]
        ylist = 2*[y]
        # for x, y in zip(xlist, ylist):
        #     letterdict[10*numPlayers+numAdaptive].append((x,y))
    elif numPlayers == 3:
        if numAdaptive == 0:
            delta = -d
        elif numAdaptive == 1:
            delta = -d/3.
        elif numAdaptive == 2:
            delta = d/3.
        elif numAdaptive == 3:
            delta = d

        xlist = [delta + x - xdelta/2., delta + x, delta + x + xdelta/2.]
        ylist = [y - ydelta*0.2886, y + ydelta/2., y - ydelta*0.2886]
        # ylist = 3*[y]
        # for x, y in zip(xlist, ylist):
        #
        #     letterdict[10*numPlayers+numAdaptive].append((x,y))

    return list(zip(xlist, ylist))


def drawTotalAdaptive(query):
    global hardcoded_designs, seed1, seed2, my_dpi, datapath
    # print(hardcoded_designs)
    totalcash_dict = {}
    # divider = 1000000.
    ydelta_dict = {10: 0.58575022518545405, 600: 0.9811286044285239, 1200: 2.111500681313383}
    xdelta = 0.11

    fig = plt.figure(figsize=(10, 5), dpi=my_dpi)
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    k = 0.5
    all_legends = []
    letter_dict = defaultdict(list)
    cost_color_dict = {600: 'b', 1200: 'm', 0: 'r'}
    cost_marker_dict = {600: 'H', 1200: '*', 0: '^'}


    totalcash_tuple_dict = defaultdict(list)
    color_dict = defaultdict(str)
    markerdict = defaultdict(str)
    order_dict = {'AA-AAA': 1, 'AAN': 2, 'AN-ANN': 3, 'NN-NNN': 4}
    design_point_dict = defaultdict(set)
    # base_letter_dict = {(0, "A"): 'A', (600, "A"): "A", (1200, "A"): 'A', (0, "N"): 'Z', (600, "N"): "S", (1200, "N"): 'O'}
    letter_marker_dict = {-2: ('s', 'k', '$c_f$: $adaptive$'),
                          10: (cost_marker_dict[0], cost_color_dict[0], '$c_f = \epsilon$'),
                          600: (cost_marker_dict[600], cost_color_dict[600], '$c_f = 0.6$'),
                          1200: (cost_marker_dict[1200], cost_color_dict[1200], '$c_f > 1$')
                          }
    design_point_dict = defaultdict(list)
    all_points = []

    objdict = pickle.load(open(datapath + 'Design_objectdict.p', 'rb'))

    for basecost in [10, 600, 1200]:
        for i, des in enumerate(hardcoded_designs):
            numPlayers = 2
            if '3.' in des:
                numPlayers = 3
            # query['elementlist'] = des
            # print(query)
            # for sgl in costrange:
            # test = 'total adaptive'
            # allfops = list(fopsGenTotal(des))
            names = ['D%d_%s%s' % (i, 'N', 'N'), 'D%d_A%s' % (i, 'N'), 'D%d_AA' % i]

            # if numPlayers == 2:
            for numA in range(numPlayers + 1):
                fopslist = list(fopsGenTotal(numPlayers, numA, basecost))
                # print(numA, fopslist)
                cashlist = []
                for fops in fopslist:
                    fops_key = ' '.join(fops)
                    Obj = objdict[des][fops_key]
                    # query['fops'] = json.dumps(fops)

                    # results = avgQueryResults(query)
                    # print(i, des, results)

                    # totalcash = sum([e[1] for e in results])/divider
                    totalcash = sum([e/divider for e in Obj.totalcash])/len(Obj.totalcash)
                    cashlist.append(totalcash)

                y = sum(cashlist) / len(fopslist)
                totalcash_dict[(basecost, i, numPlayers, numA)] = y

                values = createPoints(numPlayers, numA, i, y, xdelta, xdelta * 2.2, k)  # ydelta_dict[basecost])
                # values = [e for l in tempdict for e in l]
                # print("values:", values)
                avgx = sum([e[0] for e in values]) / len(values)
                avgy = sum([e[1] for e in values]) / len(values)
                all_points.append((avgx, avgy))
                # print(numA, basecost, i, [round(avgx, 2), round(avgy, 2)])
                design_point_dict[(basecost, i)].append(tuple([round(avgx, 2), round(avgy, 2)]))

                for j, value in enumerate(values):
                    if j<numA:
                        letter_dict[-2].append(value)
                    else:
                        letter_dict[basecost].append(value)

    print("points:",design_point_dict)

    for tup, points in sorted(list(design_point_dict.items())):
        # print(points)
        # print(tup, points)
        plt.plot(*zip(*points), 'k--', zorder=-3)

                # plt.scatter(x,y, color = color)

    plt.scatter(*zip(*all_points), marker='o', s=400, facecolors='w', edgecolors='k', zorder=-2,
                linewidths='1')

    print("first part")

    '''
        design_point_dict = defaultdict(list)
        for D, y in totalcash_dict.items():
            # print(D,y)
            x = int(re.search(r'D(\d+)_(.+)', D).group(1)) + 1
            label = re.search(r'D(\d+)_(.+)', D).group(2)
            # print(label)
            # plt.annotate(label, xy=(x, y), xycoords='data', textcoords='offset points')
            # label_dict = ({'AA': r'$\clubsuit \clubsuit$', 'AN': r'$\clubsuit \blacksquare$',
            #                'NN':r'$\blacksquare \blacksquare$', 'AAA': r'$\clubsuit \clubsuit \clubsuit$',
            #                'ANN': r'$\clubsuit \blacksquare \blacksquare$', 'NNN':r'$\blacksquare \blacksquare \blacksquare$', 'AAN': r'$\clubsuit \clubsuit \blacksquare$'})
            # label2 = label_dict[label]
            # print(label2)
            # plt.text(x, y, ha="center", va="center", s = label2)
            xdelta = 0.14
            tempdict = createPoints(label, x, y, xdelta, xdelta*2.2, k)#ydelta_dict[basecost])
            values = [e for l in tempdict.values() for e in l]
            avgx = sum([e[0] for e in values]) / len(values)
            avgy = sum([e[1] for e in values]) / len(values)
            all_points.append((avgx, avgy))
            design_point_dict[x].append((round(avgx,2), round(avgy,2)))
            print(x, avgx, avgy)
            for l in tempdict:
                # print(l)
                l2 = base_letter_dict[(basecost, l)]
                letter_dict[l2] = letter_dict[l2].union(set(tempdict[l]))


            if label.count('A') == 0:
                lab = 'NN-NNN'
                color_dict[lab] = 'b'
                markerdict[lab] = '*'
                totalcash_tuple_dict[lab].append((x, y))

            elif label.count('A') == len(label):
                lab = 'AA-AAA'
                color_dict[lab] = 'k'
                markerdict[lab] = 's'
                totalcash_tuple_dict[lab].append((x, y))
            elif label.count('A') >= 2:
                lab = 'AAN'
                color_dict[lab] = 'g'
                markerdict[lab] = '^'
                totalcash_tuple_dict[lab].append((x, y))
            else:
                lab = 'AN-ANN'
                color_dict[lab] = 'r'
                markerdict[lab] = 'o'
                totalcash_tuple_dict[lab].append((x, y))

                # plt.scatter(x,y, color = color)
        legends = []
        # for label, points in sorted(totalcash_tuple_dict.items(), key = lambda x: order_dict[x[0]]):
        #     legends.append(label)
        #     plt.scatter(*zip(*points), color = color_dict[label], marker = markerdict[label], s = 40)
        #
        # plt.legend(legends, frameon=False,ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.15), labelspacing=2)

        # for letter, points in letter_dict.items():
        #     marker, color, legend = letter_marker_dict[letter]
        #     plt.scatter(*zip(*points), marker=marker, color=color, s=k*30, linewidths='2')
        #     legends.append(legend)
        #
        # plt.legend(legends)

        # for i in range(len(hardcoded_designs) - 1):
        #     plt.axvline(i + 1.5, color='k', linestyle=':', linewidth=0.6)
        #
        for d, points in design_point_dict.items():
            print(points)
            plt.plot(*zip(*points), 'k--', zorder = -3)

        plt.scatter(*zip(*all_points), marker='o', s=k * 400, facecolors='w', edgecolors='k', zorder=-2,
                        linewidths='1')
            # plt.xlim(0.5, len(hardcoded_designs) + 0.5)
        # # print("x lim and y lim:", ax.get_ylim(), ax.get_xlim())
        # xdelta = ax.get_xlim()[1] - ax.get_xlim()[0]
        # ydelta = ax.get_ylim()[1] - ax.get_ylim()[0]
        # # ydelta_dict[basecost] = ydelta/xdelta
        # plt.ylabel('total cash (M$)')
        # # plt.title('cost functions: $N = %d, A= adaptive$'%basecost)
        # xtickDict = {0: 'I', 1: 'II', 2: 'III', 3: 'IV', 4: 'V', 5: 'VI', 6: 'VII', 7: 'VIII', 8: 'XI', 9: 'X'}
        # xticklist = ['Design %s' % xtickDict[i] for i in range(len(hardcoded_designs))]
        # plt.xticks(range(1, len(hardcoded_designs) + 1), xticklist, rotation=0)
        # plt.savefig("Total_revenue_CostFunctions_Default%s.pdf" % str(basecost).zfill(4), bbox_inches='tight')
    '''
    legends = []
    lines = []
    for k, points in letter_dict.items():
        print(k)
        marker, color, legend = letter_marker_dict[k]
        newline = plt.scatter(*zip(*points), marker=marker, color=color, s=30, linewidths='2')
        lines.append(newline)
        legends.append(legend)

    plt.legend(lines, legends, ncol = 2)

    for i in range(len(hardcoded_designs) - 1):
        print(i)
        plt.axvline(i + 1.5, color='k', linestyle=':', linewidth=0.6)


    # plt.show()
    plt.xlim(0.5, len(hardcoded_designs) + 0.5)
    # print("x lim and y lim:", ax.get_ylim(), ax.get_xlim())
    xdelta = ax.get_xlim()[1] - ax.get_xlim()[0]
    ydelta = ax.get_ylim()[1] - ax.get_ylim()[0]
    # ydelta_dict[basecost] = ydelta/xdelta
    plt.ylabel('collective value')
    # plt.title('cost functions: $N = %d, A= adaptive$'%basecost)
    xtickDict = {0: 'I', 1: 'II', 2: 'III', 3: 'IV', 4: 'V', 5: 'VI', 6: 'VII', 7: 'VIII', 8: 'XI', 9: 'X'}
    xticklist = ['Design %s' % xtickDict[i] for i in range(len(hardcoded_designs))]
    plt.xticks(range(1, len(hardcoded_designs) + 1), xticklist, rotation=0)
    # plt.savefig("Total_revenue_CostFunctions_Default.pdf", bbox_inches='tight')
    plt.savefig("Total_revenue_CostFunctions_Default.eps", format = 'eps', dpi = my_dpi, bbox_inches='tight')

    # print(ydelta_dict)
    # plt.show()


def drawStorage(docslist, design, query):
    # fopslist = [d["fops"] for d in docslist]
    # print(docslist)
    storageset = sorted(list(set([d['storagePenalty'] for d in docslist])))
    # storageset = [-1, 0, 100, 300, 500]
    # storageset = [-2, -1]
    # plt.figure()
    storage_cashlist_dict = defaultdict(list)
    for s in storageset:
        # print("storage: ", s)
        tempdocs = [d for d in docslist if int(d["storagePenalty"]) == s]
        costlsit = [d['costSGL'] for d in tempdocs]
        resultlist = [json.loads(d["results"]) for d in tempdocs]
        cashlist = [sum([e[1] for e in r]) / 1000000. for r in resultlist]
        storage_cashlist_dict[s] =  [e[1] for e in sorted(list(zip(costlsit, cashlist)))]

    storage_residual_dict = defaultdict(int)
    baseline = storage_cashlist_dict[400]
    # print("base line 400:", baseline)
    # print(maxlist)
    for s in storageset:
        cashlist = storage_cashlist_dict[s]
        # print(s, ' cash list:', cashlist)
        residual = [100*(b-a)/a for a,b in zip(baseline, cashlist)]
        # residual = [b for a,b in zip(baseline, cashlist)]
        storage_residual_dict[s] = sum(residual)

    # print(storage_residual_dict)
    return storage_residual_dict

def calResidual(docslist, design, query):
    # fopslist = [d["fops"] for d in docslist]
    print(docslist)
    storageset = sorted(list(set([d['storagePenalty'] for d in docslist])))
    # storageset = [-1, 0, 100, 300, 500]
    # storageset = [-2, -1]
    # plt.figure()
    storage_cashlist_dict = defaultdict(list)
    for s in storageset:
        # print("storage: ", s)
        tempdocs = [d for d in docslist if int(d["storagePenalty"]) == s]
        print("length of tempdocs:", len(tempdocs))
        costlsit = [d['costSGL'] for d in tempdocs]
        # resultlist = [json.loads(d["results"]) for d in tempdocs]
        cashlist = [d["cashlist"] for d in tempdocs]
        storage_cashlist_dict[s] =  [e[1] for e in sorted(list(zip(costlsit, cashlist)))]
        print(s)
        print(len(storage_cashlist_dict[s]))
        print([len(a) for a in storage_cashlist_dict[s]])

    storage_residual_dict = defaultdict(list)
    baseline = [sum(a)/float(len(a)) for a in storage_cashlist_dict[400]]
    # print("base line 400:", baseline)
    # print(maxlist)
    for s in storageset:
        cashlist = storage_cashlist_dict[s]
        # print(s, ' cash list:', cashlist)
        residual = [[100*(b-a)/a for b in l] for a,l in zip(baseline, cashlist)]
        # residual = [b for a,b in zip(baseline, cashlist)]
        print("length of residual:", len(residual))
        storage_residual_dict[s] = residual

    # print(storage_residual_dict)

    return storage_residual_dict


def runQuery(db, query, test):
    global design_dict
    residual_dict = defaultdict(list)
    N = len(design_dict)
    numseeds = 30
    boxplot_dict = {}
    storage_dict = {400: 0, 800: 1, -1: 2}
    for des, i in sorted(design_dict.items(), key = lambda x: x[1]):
        query['elementlist'] = des
        query['numTurns'] = 240
        templist = []
        for fops in fopsGen(des, test):
            query['fops'] = json.dumps(fops)
            # print("query :", query['elementlist'])
            docsresult = list(db.results.find(query))
            sample = docsresult[0]
            # print(len(docsresult), len([d['numTurns'] for d in docsresult]))
            resultlist = sorted([(d['seed'], json.loads(d["results"])) for d in docsresult])
            cashlist = [sum([e[1] for e in r[1]]) / 1000000. for r in resultlist]
            row = {k: sample[k] for k in ['fops']}
            row['cashlist'] = cashlist[:numseeds]
            costsgl, storage, auctioneer = fops2costs(row['fops'])
            if costsgl not in [600, -3]:
                continue

            row['costSGL'] = costsgl
            row['costISL'] = costsgl
            row['storagePenalty'] = storage
            # print("row:", row)
            x = i * 3 + storage_dict[storage]
            print(i, des, sum(cashlist))
            boxplot_dict[x] = cashlist
            # templist.append(row)
            # if list(db.results.find(query)):
            #     templist.append(list(db.results.find(query))[0])
            # else:
            #     print(fops)
            #     termslist = [re.search(r"(x.+),([-\.\d]+),(.+)", f) for f in fops]
            #     newfops = ["%s,%1.2f,%s"%(terms.group(1), int(terms.group(2)), terms.group(3)) for terms in termslist]
            #     query['fops'] = json.dumps(newfops)
            #     print(query)
            #     templist.append(list(db.results.find(query))[0])


        # templist2 = []
        # # fopslist = [d['fops'] for d in templist]
        # for row in templist:
        #     # print("row: ", row)
        #     fops = row['fops']
        #     costsgl, storage, auctioneer = fops2costs(fops)
        #     if costsgl not in [600, -3]:
        #         continue
        #
        #     row['costSGL'] = costsgl
        #     row['costISL'] = costsgl
        #     row['storagePenalty'] = storage
        #     templist2.append(row)
        # print(len(templist))
        # storage_residual_dict = calResidual(templist, design=design_dict[des], query = query)
        # print("Storage residual dict:", len(storage_residual_dict), storage_residual_dict)
        # for s, v in storage_residual_dict.items():
        #     print(s)
        #     print(v)
        #     residual_dict[s].append(v)

    # print(boxplot_dict)
    typ = 'Stochastic' if 'stochastic' in test else 'Deterministic'
    xstick = list(range(1,16))
    xstick_minor = [2, 5, 8, 11, 14]
    xstick_design = ['Design %s'%s for s in ['I', 'II', 'III', 'IV', 'V']]
    xstick_storagepenalty = 5 * [400, 800, 'Marginal']
    xlines = [3.5, 6.5, 9.5, 12.5]

    # print(len(boxplot_dict))
    boxplot_list = [b[1] for b in sorted(boxplot_dict.items())]
    # print(boxplot_list)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.boxplot(boxplot_list, whis = 3)
    if typ == 'Stochastic':
        for i in range(1, 6):
            j = 3*i-1
            print(xstick_design[i-1])
            # print("800:", stats.f_oneway(boxplot_list[j], boxplot_list[j-1]))
            # print("400:", stats.f_oneway(boxplot_list[j], boxplot_list[j-2]))

            print("std:", np.std(boxplot_list[j-2]), np.std(boxplot_list[j-1]), np.std(boxplot_list[j]))
            print("mean:", np.mean(boxplot_list[j-2]), np.mean(boxplot_list[j-1]), np.mean(boxplot_list[j]))


    plt.xticks(xstick, xstick_storagepenalty,  rotation = 60)

    # plt.xticks(xstick_minor, xstick_design, minor = True)
    plt.ylabel('federation value (000)')
    for xline in xlines:
        plt.axvline(xline, color='k', linestyle='-', linewidth=0.3)

    plt.xlabel('Storage Penalty')
    ax2 = plt.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks([a for a in xstick_minor])
    ax2.set_xticklabels(xstick_design)

    # plt.savefig("storagepenalty_%s.pdf" %(typ), bbox_inches='tight')
    plt.show()
    return boxplot_dict

def draw_Dictionary(residual_dict):
    plt.figure()
    legends = deque()
    dic = {e: 'SP:%d' % e for e in residual_dict if e >= 0}
    dic[-1] = 'SP:Marginal'
    dic[-2] = 'SP:QL'
    dic[-3] = 'SP:random'
    # dic[400] = 'SP: Fixed'
    dic[1200] = 'SP>1000'
    dic[0] = 'SP: Collaborative'
    baselist = [-1, 400, 800]
    # xy = sorted(zip(*[residual_dict[e] for e in baselist], list(range(len(residual_dict[-2])))), reverse=True, key=lambda x: x[1])
    xy = zip(*[residual_dict[e] for e in baselist], list(range(len(residual_dict[-1]))))
    legends = [dic[e] for e in baselist]
    Y = list(zip(*xy))
    designs = Y[-1]
    # print(designs)
    ls = iter(['--', ':', '-.', '-'])
    for s, y in zip(baselist, Y[:-1]):
        if s == 0:
            continue
        if s == -1:
            plt.scatter(range(len(y)), y, alpha = 0.5, color = 'k', marker = 's', label = dic[s], s = 80)
            # plt.plot(y, marker='o')
        elif s == 400:
            plt.scatter(range(len(y)), y, color = 'm', alpha = 0.5, marker = 'v', label = dic[s], s = 80)
            # plt.plot(y, marker='s')
        else:
            plt.scatter(range(len(y)), y, color = 'g', alpha=0.5, marker='o', label=dic[s], s = 80)


    xtickDict = {0: 'I', 1: 'II', 2:'III', 3:'IV', 4:'V', 5:'VI', 6:'VII', 7:'VIII', 8:'XI', 9:'X'}
    xticklist = ['Design %s'%xtickDict[i] for i in list(designs)]
    plt.xticks(list(range(len(residual_dict[-1]))),xticklist, rotation = 0)
    plt.legend(legends)
    for i in range(len(residual_dict[-1])-1):
        plt.axvline(i + 0.5, color='k', linestyle='-', linewidth=0.3)
    plt.xlim(-0.5, len(residual_dict[-1])-0.5)

def sumDics(db, query, test):
    typ = 'Stochastic' if 'stochastic' in test else 'Deterministic'
    residual_dict = defaultdict(list)
    # residual_dict = runQuery(db, query)

    for capacity, links in [(2,2)]:
        query['capacity'] = capacity
        query['links'] = links
        # print('Sum Dics query:')
        # print(query)
        tempdict = runQuery(db, query, test)
        print(len(tempdict))
        print([type(t) for t in tempdict])

        # for s in tempdict:
        #     # print("temp dict:", s)
        #     # print("tempdict seed:", s)
        #
        #     if s in residual_dict:
        #         residual_dict[s] = [a+b for a,b in zip(residual_dict[s], tempdict[s])]
        #     else:
        #         residual_dict[s] = tempdict[s]
        #
        #     print('residual dict s:', residual_dict[s])

        # draw_Dictionary(tempdict)
        # # plt.title('storage:%d, links:%d' % (capacity, links))
        # plt.ylabel('Improvement over baseline (%)')
        #
        #
        # plt.savefig("%s_storagepenalty_%d_%d.pdf" % (typ, capacity, links), bbox_inches='tight')
def avgQueryResults(query):
    # receives a query and calculates results as the average of all seeds (global variables)
    global db, seed1, seed2
    resultslist = []
    for seed in range(seed1, seed2):
        query['seed'] = seed
        # print(query)
        print(query)
        resultslist.append(json.loads(list(db.results.find(query))[0]['results']))

    final = resultslist[0]
    for i, _ in enumerate(final):
        # print(i, [len(r) for r in resultslist])
        final[i][1] = sum([r[i][1] for r in resultslist])/(seed2 - seed1)

    # print(final)
    return final

def avgBidingList(query):
    # receives a query and calculates results as the average of all seeds (global variables)
    global db, seed1, seed2
    biddinglist = []
    for seed in range(seed1, seed2):
        query['seed'] = seed
        # print(query)
        # print(query)
        templist = json.loads(list(db.results.find(query))[0]['biddings'])
        # print(len(templist))
        biddinglist.append(templist)
        # print(biddinglist[-1])

    final = []
    for i in range(len(biddinglist[0])):
        templist = [b[i] for b in biddinglist]
        final.append([sum(e)/(seed2-seed1) for e in zip(*templist)])

    return final

def avgSharedLinks(query):
    global db, seed1, seed2
    pricelist = []
    sharedlist = []
    for seed in range(seed1, seed2):
        query['seed'] = seed
        # print(query)
        templinks = json.loads(list(db.results.find(query))[0]['sharedlinks'])
        # print("len of temp list:", templist)
        sharedlist.append(templinks)

    finaldict = defaultdict(int)
    for i in range(len(sharedlist[0])):
        tempshared = [b[i] for b in sharedlist]
        countdict = defaultdict(int)
        temptotal = chain(*tempshared)
        for tup in temptotal:
            countdict[int(tup[0])] += 1
            finaldict[int(tup[0])] += int(tup[1])

        finaldict = {k: v/countdict[k] for k in finaldict}

    return sorted(finaldict.items())




def avgAucPrices(query):
    # receives a query and calculates results as the average of all seeds (global variables)
    global db, seed1, seed2
    pricelist = []
    sharedlist = []
    for seed in range(seed1, seed2):
        query['seed'] = seed
        # print(query)
        templist = json.loads(list(db.results.find(query))[0]['timepricelist'])
        templinks = json.loads(list(db.results.find(query))[0]['sharedlinks'])
        # print("len of temp list:", templist)
        pricelist.append(templist)
        sharedlist.append(templinks)



    final = []
    # pricedict = defaultdict(int)
    for i in range(len(pricelist[0])):
        pricedict = defaultdict(int)
        countdict = defaultdict(int)

        templist = [b[i] for b in pricelist]
        tempshared = [b[i] for b in sharedlist]
        shareddict = defaultdict(int, tempshared[0])
        for li in templist:

            for tup in li:
                if shareddict[tup[0]]>=1:
                    pricedict[int(tup[0])] += shareddict[tup[0]]*min(tup[1], 1000)
                    countdict[int(tup[0])] += shareddict[tup[0]]
                # if tup[1]>999:
                #     print("federate, price and links:", i+1, tup[1], shareddict[tup[0]])

        for time, price in pricedict.items():
            pricedict[time] = price/countdict[time]
            # print(pricedict[time])

        final.append(pricedict)

    return final


def drawAdaptiveSGL(query, test):
    global divider, my_dpi, objdict, epsilon
    federatecash_dict_list1 = []
    federatecash_dict_list2 = []

    totalcash_dict_list = []

    for des in hardcoded_designs:
        query['elementlist'] = des
        numPlayers = 2
        if '3.' in des:
            numPlayers = 3

        federate_dict_1 = {}
        federate_dict_2 = {}
        totalcash_dict = {}
        # fopslist = list(fopsGen(des, test))
        for fops, fops_adaptive in fopsGen(des, test):
            print("fops:", fops, fops_adaptive)
            query['fops'] = fops
            # print(query)
            sgl = int(re.search(r"x([-\d]+),.+", ' '.join(fops)).group(1))
            # print("length of query:", list(db.results.find(query)))
            # docs = list(db.results.find(query))[0]
            # results = avgQueryResults(query)
            # federatecash_1 = sum([e[1] for e in results])/len(results)
            Obj = objdict[des][' '.join(fops)]
            federatecash_1 = sum([e for e in Obj.federatecash])/len(Obj.federatecash)

            cashlist_2 = []
            cashlist_a2 = []
            for n in range(numPlayers):
                tempfops = fops_adaptive
                temp = tempfops[0]
                tempfops[0] = tempfops[n]
                tempfops[n] = temp
                # query['fops'] = json.dumps(tempfops)
                # results_adaptive = avgQueryResults(query)
                Obj_adaptive = objdict[des][' '.join(tempfops)]
                # cashlist_a2.append(results_adaptive[n][1])
                # print(len(Obj_adaptive.federatecash))
                cashlist_a2.append(Obj_adaptive.federatecash[n])
                cashlist_2.extend([r for i, r in enumerate(Obj_adaptive.federatecash) if i!=n])

            # print(cashlist_2)
            # print(cashlist_a2)
            federatecash_2 = sum(cashlist_2)/float(len(cashlist_2))
            federatecash_a2 = sum(cashlist_a2)/float(len(cashlist_a2))
            # print(query)
            # print("length of query:", list(db.results.find(query)))
            # docs_a = list(db.results.find(query))[0]

            # results = json.loads(docs['results'])
            # results = avgQueryResults(query)
            # results_adaptive = json.loads(docs_a['results'])
            # federatecash_a1 = results_adaptive[0][1]
            # federatecash_2 = sum([e[1] for e in results[1:]])/len(results[1:])
            # federetecash_a2 = sum([e[1] for e in results_adaptive[1:]])/len(results_adaptive[1:])

            # totalcash = sum(Obj.totalcash)
            # totalcash_adaptive = sum(Obj_adaptive.totalcash)
            # totalcash_adaptive = sum(federatecash_2)
            print("Federate cash:", federatecash_1, federatecash_2, federatecash_a2)
            federate_dict_1[sgl] = (federatecash_1, federatecash_a2)
            federate_dict_2[sgl] = (federatecash_1, federatecash_2)
            # print(federatecash_1, federatecash_a1)
            # print(federatecash_2, federetecash_a2)
            # totalcash_dict[sgl] = (totalcash, totalcash_adaptive)

        federatecash_dict_list1.append(federate_dict_1)
        federatecash_dict_list2.append(federate_dict_2)
        totalcash_dict_list.append(totalcash_dict)

    xtickDict = {0: 'I', 1: 'II', 2:'III', 3:'IV', 4:'V', 5:'VI', 6:'VII', 7:'VIII', 8:'XI', 9:'X'}
    xticklist = ['Design %s'%xtickDict[i] for i in range(len(hardcoded_designs))]

    delta = 0.3
    marker_dict = {-3: 'v', epsilon: '^', 600: 'H', 1200: '*' ,'adaptive': 's'}
    color_dict = {-3: 'g', epsilon: 'r', 600: 'b', 1200: 'm' ,'adaptive': 'k'}
    function_dict = {-3: '$c_f$=tri-random', epsilon: '$c_f=\epsilon$', 600: '$c_f= 0.6$', 1200: '$c_f>1$' ,'adaptive': '$c_f:adaptive$'}
    order_dict = {-3: 4.5, epsilon: 2, 600: 3, 1200: 4 ,'adaptive': 5}
    sp_list = [epsilon, 600, 1200]

    all_points1 = defaultdict(list)
    all_points_adaptive1 = defaultdict(list)
    all_edges1 = defaultdict(list)
    all_points2 = defaultdict(list)
    all_edges2 = defaultdict(list)
    all_points_adaptive2 = defaultdict(list)

    for i, cash_dict in enumerate(federatecash_dict_list1):
        for k, v in cash_dict.items():
            print("adaptive cash: ", v)
            point1 = (i+1-delta, v[0]/divider)
            point2 = (i+1+delta, v[1]/divider)
            all_points1[k].append(point1)
            # all_points1['adaptive'].append(point2)
            all_points_adaptive1[k].append(point2)
            all_edges1[k].append((point1, point2))

    for i, cash_dict in enumerate(federatecash_dict_list2):
        for k, v in cash_dict.items():
            print("nonadaptive cash: ", v)
            point1 = (i+1-delta, v[0]/divider)
            point2 = (i+1+delta, v[1]/divider)
            all_points2[k].append(point1)
            # all_points2['adaptive'].append(point2)
            all_points_adaptive2[k].append(point2)
            # all_points2['adaptive'].append(point2)
            all_edges2[k].append((point1, point2))

    legends = []
    lines = []

    for s in sp_list:
        fig = plt.figure()
        ax1 = fig.add_axes([0.1, 0.5, 0.9, 0.35])
        points = all_points1[s]
        legends = []
        lines = []
        # for s, points in sorted(all_points.items(), key = lambda x: order_dict[x[0]]):
        print(points)
        lines.append(ax1.scatter(*zip(*points), marker = marker_dict[s], color = color_dict[s], s = 100, facecolors = color_dict[s], linewidth='2'))
        legends.append(function_dict[s])

        points = all_points_adaptive1[s]
        lines.append(ax1.scatter(*zip(*points), marker = marker_dict['adaptive'], color = 'k', s = 80, facecolors = 'k', linewidth='2'))
        legends.append(function_dict['adaptive'])
        plt.legend(legends, loc = 'best')
        # fig.legend(lines, legends, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.4, 1.2), labelspacing=2)

        for edge in all_edges1[s]:
            # plt.plot(*zip(*edge), 'k:', linewidth = 0.7)
            ax1.arrow(0.1*edge[1][0]+0.9*edge[0][0], 0.1*edge[1][1]+0.9*edge[0][1], 0.7*(edge[1][0]-edge[0][0]), 0.7*(edge[1][1]-edge[0][1]), head_width=0.08, head_length=0.1, linewidth = 0.7, fc ='k', ec = 'k', zorder = -1)

        plt.xticks(range(1, len(hardcoded_designs)+1), ['' for i in xticklist], rotation = 0)

        for i in range(len(hardcoded_designs)-1):
            ax1.axvline(i+1.5, color = 'k', linestyle = '-', linewidth = 0.3)

        plt.ylabel('adaptive federate')
        plt.xlim(0.5, len(hardcoded_designs)+0.5)
        # plt.ylim(0.5,4.1)
        ax2 = fig.add_axes([0.1, 0.1, 0.9, 0.35])

        # if s in all_edges2:
        points = all_points2[s]
        # print(s, points)
        # for s, points in sorted(all_points2.items(), key = lambda x: order_dict[x[0]]):
        lines.append(ax2.scatter(*zip(*points), marker = marker_dict[s], color = color_dict[s], s = 100, facecolors = color_dict[s], linewidth='2'))
        legends.append(function_dict[s])

        # elif s in all_points_adaptive:
        points = all_points_adaptive2[s]
        # print(s, points)
        # for s, points in sorted(all_points_adaptive.items(), key = lambda x: order_dict[x[0]]):
        lines.append(ax2.scatter(*zip(*points), marker = marker_dict[s], color = color_dict[s], s = 100, facecolors = color_dict[s], linewidth='2'))
        # legends.append(function_dict[s])

        # edge = all_edges2[s]
        for edge in all_edges2[s]:
            # plt.plot(*zip(*edge), 'k:', linewidth = 0.7)
            ax2.arrow(0.1*edge[1][0]+0.9*edge[0][0], 0.1*edge[1][1]+0.9*edge[0][1], 0.7*(edge[1][0]-edge[0][0]), 0.7*(edge[1][1]-edge[0][1]), head_width=0.08, head_length=0.1, linewidth = 0.7, fc ='k', ec = 'k', zorder = -1)

        plt.xticks(range(1, len(hardcoded_designs)+1), xticklist, rotation = 0)
        for i in range(len(hardcoded_designs)-1):
            ax2.axvline(i+1.5, color = 'k', linestyle = '-', linewidth = 0.3)

        # plt.ylim(0.5,4.1)
        # fig.legend(lines[:2]+lines[3:], legends[:2]+legends[3:], loc='best')#, ncol = 2)
        plt.ylabel('non-adaptive federate')
        # plt.savefig("Federate_revenue_costfunction_V3_sp%s.pdf"%str(s), bbox_inches='tight')
        plt.savefig("Federate_revenue_costfunction_V3_sp%s.eps"%str(s), format='eps', dpi=my_dpi, bbox_inches='tight')

    plt.show()

    # print(federatecash_dict_list1)
    # print(totalcash_dict_list)


def drawStoragePenalty(db):
    query = {'experiment': 'Storage Penalty V2'}
    test = 'regular storage deterministic'
    sumDics(db, query, test)

    test = 'regular storage stochastic'
    sumDics(db, query, test)
    plt.show()

def drawFederateAdaptive(db=None):
    global numTurns
    query = {'experiment': 'Adaptive Cost V2', 'capacity': 2, 'links': 2, 'numTurns': numTurns}
    test = 'federate adaptive'

    drawAdaptiveSGL(query, test)

# def drawTotalAdaptive(db):
#     query = {'experiment': 'Adaptive Cost', 'capacity': 2, 'links': 2}
#     drawTotalAdaptive(query)

def drawAdaptiveAuctioneer(query):
    global db, design_dict, xticklist, my_dpi, fops_dicts
    # query = {'capacity': 2, 'links': 2, 'numTurns': 2400}
    totalcash_dict = defaultdict(list)
    all_points1 = []
    all_points2 = []
    all_edges = []
    test = 'auctioneer'
    # divider = 1000000
    all_federate_edges = []
    objdict = pickle.load(open(datapath + 'Design_objectdict.p', 'rb'))

    fig1 = plt.figure(1, figsize=(15,  6), dpi=my_dpi)
    fig2 = plt.figure(2, figsize=(15,  6), dpi=my_dpi)

    for j, des in enumerate(hardcoded_designs):
        if j<=1:
            numPlayers = 2
        else:
            numPlayers = 3

        fops_adaptive = fops_dict[numPlayers]['adaptive']
        fops_auctioneer = fops_dict[numPlayers]['auctioneer']

        obj1 = objdict[des][fops_adaptive]
        obj2 = objdict[des][fops_auctioneer]
        totalcash1 = sum(obj1.federatecash)
        totalcash2 = sum(obj2.federatecash)
        fc1 =  obj1.federatecash#[sum(e) for e in obj1.cashes]
        fc2 =  obj2.federatecash#[sum(e) for e in obj2.cashes]
        points1 = [(1+design_dict[des]-0.3, e/divider) for e in fc1]
        points2 = [(1+design_dict[des]+0.3, e/divider) for e in fc2]
        #
        # for _, fops_adaptive, fops_auctioneer in fopsGen(des, test):
        #     query['fops'] = json.dumps(fops_adaptive)
        #     query['experiment'] = 'Adaptive Cost V4'
        #     # print(query)
        #     results1 = avgQueryResults(query)
        #     # docs_adaptive = list(db.results.find(query))[0]
        #     query['fops'] = json.dumps(fops_auctioneer)
        #     # query['experiment'] = 'Adaptive Cost Auctioneer'
        #     # print(query)
        #     # docs_auctioneer = list(db.results.find(query))[0]
        #
        #     # results1 = avgQueryResults(query)#json.loads(docs_adaptive['results'])
        #     results2 = avgQueryResults(query)#json.loads(docs_auctioneer['results'])
        #
        #     totalcash1 = sum([e[1] for e in results1])
        #     totalcash2 = sum([e[1] for e in results2])
        #
        #     points1 = [(1+design_dict[des]-0.3, e[1]/divider) for e in results1]
        #     points2 = [(1+design_dict[des]+0.3, e[1]/divider) for e in results2]

        all_points1.extend(points1)
        all_points2.extend(points2)

        all_federate_edges.extend(list(zip(points1, points2)))


        point1 = (1+design_dict[des]-0.3, totalcash1/divider)
        point2 = (1+design_dict[des]+0.3, totalcash2/divider)


        print("collective value :", point1, point2)
        totalcash_dict['adaptive'].append(point1)
        totalcash_dict['auctioneer'].append(point2)
        all_edges.append((point1, point2))


    print(totalcash_dict)
    # fig = plt.figure()
    # plt.figure(figsize=(5, 6), dpi=my_dpi)
    # fig, ax = plt.subplots(1, 1, figsize=(5, 6), dpi=my_dpi)
    # fig = plt.figure(figsize=(5, 6), dpi=my_dpi)
    plt.figure(1)
    ax1 = fig1.add_axes([0.1, 0.5, 0.9, 0.35])

    ax1.scatter(*zip(*totalcash_dict['adaptive']), marker='s', color='k', s=100, facecolors='w', linewidth='2')
    ax1.scatter(*zip(*totalcash_dict['auctioneer']), marker='s', color='k', s=100, facecolors='k', linewidth='2')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.legend(['w/o Auc.', 'w/ Auc.'], loc=2)

    for i in range(len(hardcoded_designs)-1):
        plt.axvline(i+1.5, color = 'k', linestyle = '-', linewidth = 1)

    for edge in all_edges:
        # plt.plot([edge[0][0], edge[0][1]], [edge[1][0], edge[1][1]], 'k--')
        plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'k--', zorder = -1)
        # plt.plot(*zip(*edge), 'k:', linewidth = 0.7)
        # plt.arrow(edge[0][0], edge[0][1], 0.7*(edge[1][0]-edge[0][0]), 0.7*(edge[1][1]-edge[0][1]), head_width=0.3*(edge[1][0]-edge[0][0]), head_length=0.3*(edge[1][1]-edge[0][1]),
        #           linewidth = 1.5, fc ='k', ec = 'k', zorder = -1, linestyle = ':')

    plt.xticks(range(1, len(hardcoded_designs) + 1), ['' for i in xticklist], rotation=0)
    plt.ylabel("collective value")
    plt.figure(2)
    ax2 = fig2.add_axes([0.1, 0.1, 0.9, 0.35])

    ax2.scatter(*zip(*all_points1), marker='o', color='k', s=90, facecolors='w', linewidth='2')
    ax2.scatter(*zip(*all_points2), marker='o', color='k', s=90, facecolors='k', linewidth='2')
    plt.legend(['w/o Auc.', 'w/ Auc.'], loc = 'best')
    for i in range(len(hardcoded_designs) - 1):
        plt.axvline(i + 1.5, color='k', linestyle='-', linewidth=1)

    for edge in all_federate_edges:
        # print([edge[0][0], edge[0][1]], [edge[1][0], edge[1][1]])
        plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'k--', Zorder = -1)
        # plt.plot(*zip(*edge), 'k:', linewidth = 0.7)
        # plt.arrow(edge[0][0], edge[0][1], 0.7 * (edge[1][0] - edge[0][0]), 0.7 * (edge[1][1] - edge[0][1]),
        #           head_width=0.3 * (edge[1][0] - edge[0][0]), head_length=0.1 + 0.2 * (edge[1][1] - edge[0][1]), linewidth=1.5, fc='k', ec='k', zorder=-1, linestyle = ':')

    plt.ylabel("federate value")
    plt.xticks(range(1, len(hardcoded_designs) + 1), xticklist, rotation=0)
    # plt.savefig("TotalCash_Adaptive_vs_Auctioneer.pdf", bbox_inches='tight')
    fig1.savefig("TotalCash_Adaptive_vs_Auctioneer_u.eps", format='eps', dpi=my_dpi, bbox_inches='tight')
    fig2.savefig("TotalCash_Adaptive_vs_Auctioneer_l.eps", format='eps', dpi=my_dpi, bbox_inches='tight')
    # plt.show()



def drawStorageCoefficient(db):
    global xticklist, hardcoded_designs, divider
    # print(xticklist)
    query = {'experiment': 'Storage Penalty', 'capacity': 2, 'links': 2, 'numTurns': 2400}
    for j, des in enumerate(hardcoded_designs):
        # print(des)
        # print(xticklist[j])
        query['elementlist'] = des
        numPlayers = 2
        if '3.' in des:
            numPlayers = 3
        coefreslist = []
        pricereslist = []
        legends = []
        for test in ['storage stochastic', 'storage deterministic']:
            coefresulttuples = []
            priceresultsdict = {}
            for fops in fopsGen(des, test):
                # print(fops)
                query['fops'] = json.dumps(fops)
                query['experiment'] = 'Storage Penalty'
                # print(query)
                docs = list(db.results.find(query))
                # print(query, len(docs))
                docs = docs[0]
                # print("length of docs:", len(docs))
                results = json.loads(docs['results'])
                totalcash = sum([e[1] for e in results])/divider
                k = float(re.search(r'x.+,([-\.\d]+),.+', fops[0]).group(1))
                # print(k)
                if k<0:
                    coefresulttuples.append((abs(k), totalcash))
                else:
                    priceresultsdict[k] = totalcash
            coefreslist.append(coefresulttuples)
            pricereslist.append(priceresultsdict)
            legends.append(test)

        # print(coefresulttuples)
        # print(priceresultsdict)
        plt.figure()
        # coefresulttuples = sorted(coefresulttuples)
        plt.plot(*list(zip(*sorted(coefreslist[0]))))
        plt.plot(*list(zip(*sorted(coefreslist[1]))))
        stochasticMAX = max(pricereslist[0].items(), key = lambda x: x[1])
        deterministicMAX = max(pricereslist[1].items(), key = lambda x: x[1])
        plt.axhline(deterministicMAX[1], linestyle = '--', c = 'r')
        legends.append("deter-cost SP:%d"%deterministicMAX[0])
        plt.axhline(stochasticMAX[1], linestyle = '-.', c = 'b')
        legends.append("stoch-cost SP:%d"%stochasticMAX[0])
        plt.legend(legends)
        plt.title("%s"%(xticklist[j]))
        plt.ylabel('total cash')
        plt.xlabel('storage coefficient')
        plt.savefig("storagepenalty_coefficient_%s.pdf"%xticklist[j], bbox_inches='tight')

    plt.show()


def drawGraphbyDesign(number, design):
    global my_dpi
    elements = design.split(' ')
    federates = set([int(e[0]) for e in elements])
    federates_location_dict = defaultdict(list)
    federates_type_dict = defaultdict(list)
    federate_coordinates_dict = defaultdict(list)
    plt.figure(figsize=(5, 5), dpi=my_dpi)
    for r in [4, 2.25, 1.]:
        x = np.linspace(-1.0*r, 1.0*r, 50)
        y = np.linspace(-1.0*r, 1.0*r, 50)
        X, Y = np.meshgrid(x, y)
        F = X ** 2 + Y ** 2 - r
        plt.contour(X, Y, F, [0], colors='k', linewidths = 0.3, origin = 'lower', zorder = -1)

    font = FontProperties()
    font.set_style('italic')
    font.set_weight('bold')
    font.set_size('x-small')
    for x,y,lab in [(0,0,'SUR'), (0, 1, "LEO"),(0, 1.5, 'MEO'),(0, 2, 'GEO')]:
        # plt.annotate(lab, xy = (x,y), xytext = (x-0.2, y-0.1))
        plt.text(x,y, ha="center", va="center", s = lab, bbox = dict(fc="w", ec="w", lw=2),fontproperties=font)

    for i, (x, y) in enumerate([convertLocation2xy(e) for e in ['OOO'+str(i) for i in range(1,7)]]):
        plt.text(x, y, ha="center", va="center", s=str(i+1), bbox=dict(fc="none", ec="none", lw=2), fontproperties=font)

    font.set_size('medium')
    plt.text(0, 2.3 , ha="left", va="center", s=r'$|\rightarrow \theta$', bbox=dict(fc="w", ec="w", lw=2), fontproperties=font)

    types_dict = {'GroundSta': "G", 'Sat': 'S'}
    colordict = {'F1': 'yellow', 'F2': 'lightcyan', 'F3': 'lightgrey'}
    allpossiblelocations = []
    for location in ['SUR', 'LEO', 'MEO', 'GEO']:
        for i in range(1,7):
            allpossiblelocations.append(location + str(i))

    allpossiblecoordinates = [convertLocation2xy(e) for e in allpossiblelocations]
    plt.scatter(*zip(*allpossiblecoordinates), marker = "H", s = 800, color = 'k', facecolors = 'w')
    for f in federates:
        print(design, elements)
        types = [re.search(r'\d\.(.+)@(\w+\d)', e).group(1) for e in elements if '%d.' % f in e]
        federates_type_dict['F%d'%f] = [types_dict[t] for t in types]
        federates_location_dict['F%d'%f] = [re.search(r'(.+)@(\w+\d)', e).group(2) for e in elements if '%d.'%f in e]
        federate_coordinates_dict['F%d'%f] = [convertLocation2xy(loc) for loc in federates_location_dict['F%d'%f]]
        print(federate_coordinates_dict)
        plt.scatter(*zip(*federate_coordinates_dict['F%d'%f]), marker = "H", s = 800, edgecolors = 'k', facecolors = colordict['F%d'%f], linewidth='3')
        for x, y in federate_coordinates_dict['F%d'%f]:
            plt.annotate('F%d'%f, xy = (x, y), xytext = (x-0.1, y-0.075))


    plt.xticks([])
    plt.yticks([])
    rlim = 2.5
    plt.xlim(-rlim, rlim)
    plt.ylim(-rlim+0.2, rlim)
    plt.axis('off')
    des_roman_dict = {1: 'I', 2: 'II', 3:'III', 4:'IV', 5:'V', 6:'VI'}
    # plt.savefig("testDesign_%s.pdf"%des_roman_dict[number], bbox_inches='tight')
    plt.savefig("testDesign_%s.eps"%des_roman_dict[number], format='eps', dpi=my_dpi, bbox_inches='tight')

def convertLocation2radial(location):
    global radiusDict
    if 'SUR' in location:
        r = radiusDict['SUR']
    elif 'LEO' in location:
        r = radiusDict['LEO']
    elif 'MEO' in location:
        r = radiusDict['MEO']
    elif "GEO" in location:
        r = radiusDict['GEO']
    else:
        r = radiusDict['ELSE']

    sect = int(re.search(r'.+(\d)', location).group(1))
    tetha = (sect - 1) * math.pi / 3

    # print location, x, y
    return (r, tetha)


def convertradius2xy(r, tetha):
    x, y = (r * math.cos(tetha), r * math.sin(tetha))
    return (x,y)

def moveElement(r, tetha, Nsteps):
    global radiusDict
    substep = 2*math.pi/Nsteps
    if r == radiusDict['LEO']:
        return (r, (tetha + 2* substep)%(2*math.pi))
    elif r == radiusDict['MEO']:
        return (r, (tetha + substep)%(2*math.pi))
    else:
        return (r, tetha%(2*math.pi))

def ifLink(radial1, radial2):
    global radiusDict
    r1, tetha1 = radial1
    r2, tetha2 = radial2
    delta1 = abs(tetha1 - tetha2)
    delta2 = abs(2*math.pi - delta1)
    delta = min(delta1, delta2)
    result = None
    if r1 == r2 and r1 == radiusDict['SUR']:
        result = False
    elif r1 == radiusDict['SUR'] or r2 == radiusDict['SUR']:
        if delta <= math.pi/6:
            result = True
        else:
            result = False
    # elif r1 == radiusDict['GEO'] or r2 == radiusDict['GEO']:
    #     if delta <= math.pi/4:
    #         return True
    #     else:
    #         return False
    else:
        if delta <= 2*math.pi/6:
            result = True
        else:
            result = False
    # print(radial1,radial2, result)
    return result

def drawSatelliteModel(elementlist, Nsteps):
    global radialpoints, seed
    radialpoints = [convertLocation2radial(e) for e in elementlist]
    # xypoints = [convertradius2xy(r, tetha) for r, tetha in radialpoints]
    # posDict = {e: pos for e, pos in zip(elementlist, radialpoints)}
    my_dpi = 150
    # fig, ax = plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
    fig, ax = plt.subplots(1, 1, figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
    G = nx.DiGraph()
    G.add_nodes_from(elementlist)
    def update(t):
        global radialpoints, colorDict
        ax.clear()
        radialDict = {e: r for e, r in zip(elementlist, radialpoints)}
        xypoints = [convertradius2xy(r, tetha) for r, tetha in radialpoints]
        posDict = {e: pos for e, pos in zip(elementlist, xypoints)}
        # print(xypoints)
        for e in elementlist:
            node = nx.draw_networkx_nodes(G, pos = posDict, nodelist=[e], node_color=colorDict[e], node_size=200, node_shape='H', linewidths = 2, cmap=plt.get_cmap('jet'))
            node.set_edgecolor('k')

        possibleedges = [(a,b) for a, b in product(elementlist, elementlist) if (a != b) and ifLink(radialDict[a], radialDict[b])]
        # possibleedges = [(a,b) for a, b in possibleedges if ('SUR' in a or 'SUR' not in b) and (colorDict[a] == colorDict[b])]
        G.remove_edges_from(G.edges())
        G.add_edges_from(possibleedges)
        nx.draw_networkx_edges(G, pos = posDict, width = 0.5)
        lim = max(radiusDict.values())
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.axis('off')
        plt.Circle((0, 0), 1.1, color='k', fill=True)
        x = np.linspace(-lim, lim, 50)
        y = np.linspace(-lim, lim, 50)
        X, Y = np.meshgrid(x, y)
        F1 = X ** 2 + Y ** 2 - radiusDict['SUR']
        F2 = X ** 2 + Y ** 2 - radiusDict['LEO']**2
        F3 = X ** 2 + Y ** 2 - radiusDict['MEO']**2
        plt.contour(X, Y, F1, [0], zorder = -1, colors = 'k')
        plt.contour(X, Y, F2, [0], zorder = -1, linestyles = 'dotted')
        plt.contour(X, Y, F3, [0], zorder = -1, linestyles = 'dotted')
        radialpoints = [moveElement(r, tetha, Nsteps=Nsteps) for r, tetha in radialpoints]

    # for t in range(timesteps):
    #     xypoints = [convertradius2xy(r, tetha) for r, tetha in radialpoints]


    ani = matplotlib.animation.FuncAnimation(fig, update, frames=Nsteps, interval=.1, repeat=False)
    ani.save('propagation_%s.gif'%(str(seed).zfill(3)), dpi=150, writer='imagemagick')
    # plt.show()

def drawAnimativeDesignModel(elementlist, Nsteps = 6):
    global radialpoints, seed, my_dpi
    radialpoints = [convertLocation2radial(e) for e in elementlist]
    # xypoints = [convertradius2xy(r, tetha) for r, tetha in radialpoints]
    # posDict = {e: pos for e, pos in zip(elementlist, radialpoints)}
    # my_dpi = 150
    # fig, ax = plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=my_dpi)
    G = nx.DiGraph()
    G.add_nodes_from(elementlist)
    def update(t):
        global radialpoints, colors, federateDict
        federates = sorted(federateDict.values())
        federates_location_dict = defaultdict(list)
        federates_type_dict = defaultdict(list)
        federate_coordinates_dict = defaultdict(list)
        for r in [4, 2.25, 1.]:
            x = np.linspace(-1.0 * r, 1.0 * r, 50)
            y = np.linspace(-1.0 * r, 1.0 * r, 50)
            X, Y = np.meshgrid(x, y)
            F = X ** 2 + Y ** 2 - r
            plt.contour(X, Y, F, [0], colors='k', linewidths=0.3, origin='lower', zorder=-1)

        font = FontProperties()
        font.set_style('italic')
        font.set_weight('bold')
        font.set_size('x-small')
        for x, y, lab in [(0, 0, 'SUR'), (0, 1, "LEO"), (0, 1.5, 'MEO'), (0, 2, 'GEO')]:
            # plt.annotate(lab, xy = (x,y), xytext = (x-0.2, y-0.1))
            plt.text(x, y, ha="center", va="center", s=lab, bbox=dict(fc="w", ec="w", lw=2), fontproperties=font)

        for i, (x, y) in enumerate([convertLocation2xy(e) for e in ['OOO' + str(i) for i in range(1, 7)]]):
            plt.text(x, y, ha="center", va="center", s=str(i + 1), bbox=dict(fc="none", ec="none", lw=2),
                     fontproperties=font)

        font.set_size('medium')
        plt.text(0, 2.3, ha="left", va="center", s=r'$|\rightarrow \theta$', bbox=dict(fc="w", ec="w", lw=2),
                 fontproperties=font)

        types_dict = {'GroundSta': "G", 'Sat': 'S'}
        allpossiblelocations = []
        for location in ['SUR', 'LEO', 'MEO', 'GEO']:
            for i in range(1, 7):
                allpossiblelocations.append(location + str(i))

        allpossiblecoordinates = [convertLocation2xy(e) for e in allpossiblelocations]
        plt.scatter(*zip(*allpossiblecoordinates), marker="H", s=800, color='k', facecolors='w')
        for f in federates:
            federates_location_dict['F%d' % f] = [e for e in elementlist if f == federateDict[e]]
            federate_coordinates_dict['F%d' % f] = [convertLocation2xy(loc) for loc in
                                                    federates_location_dict['F%d' % f]]
            plt.scatter(*zip(*federate_coordinates_dict['F%d' % f]), marker="H", s=800, edgecolors='k',
                        facecolors=colors[f], linewidth='3')
            for x, y in federate_coordinates_dict['F%d' % f]:
                plt.annotate('F%d' % f, xy=(x, y), xytext=(x - 0.1, y - 0.075))

        plt.xticks([])
        plt.yticks([])
        rlim = 2.5
        plt.xlim(-rlim, rlim)
        plt.ylim(-rlim + 0.2, rlim)
        plt.axis('off')

    # for t in range(timesteps):
    #     xypoints = [convertradius2xy(r, tetha) for r, tetha in radialpoints]


    ani = matplotlib.animation.FuncAnimation(fig, update, frames=Nsteps, interval=.5, repeat=False)
    ani.save('desinganim_%s.gif'%(str(seed).zfill(3)), dpi=150, writer='imagemagick')
    plt.show()



# This section is for drawing final results regarding pricing and bidding adaptive agents including the auctioneer
def drawPricing():
    global axes_list_3, axes_list_5, axes_list_10, axes_list_15, window, divider, xticklist, div2, datapath, my_dpi, fops_dict
    # objlist = pickle.load(open(datapath+'Design_objectlists.p', 'rb'))
    objdict = pickle.load(open(datapath+'Design_objectdict.p', 'rb'))

    fig1 = plt.figure(1, figsize=(15,  3), dpi=my_dpi)
    fig2 = plt.figure(2, figsize=(15,  3), dpi=my_dpi)
    fig3 = plt.figure(3, figsize=(15,  3), dpi=my_dpi)
    fig4 = plt.figure(4, figsize=(15,  3), dpi=my_dpi)
    # print([len(e) for e in objlist])
    for j, des in enumerate(hardcoded_designs):
        plt.figure(1)
        if j<=1:
            numPlayers = 2
        else:
            numPlayers = 3

        fops0 = fops_dict[numPlayers]['centralized']
        fops_adaptive = fops_dict[numPlayers]['adaptive']
        fops_auctioneer = fops_dict[numPlayers]['auctioneer']

        centralObj = objdict[des][fops0]
        adaptiveObj = objdict[des][fops_adaptive]
        auctioneerObj = objdict[des][fops_auctioneer]
        # ax = fig.add_axes(axes_list_10[0][j])
        ax1 = fig1.add_axes(axes_list_5[j])
        ax2 = fig2.add_axes(axes_list_5[j])
        print(axes_list_5[j])
        # print(len(centralObj.avgbiddings), len(adaptiveObj.avgbiddings), len(auctioneerObj.avgbiddings))
        # adaptivebid25 = [e/div2 for e in movingaverage(adaptiveObj.bid25, window)]
        # adaptivebid75 = [e/div2 for e in movingaverage(adaptiveObj.bid75, window)]
        # aucbid25 = [e/div2 for e in movingaverage(auctioneerObj.bid25, window)]
        # aucbid75 = [e/div2 for e in movingaverage(auctioneerObj.bid75, window)]
        # print(adaptiveObj.query, adaptiveObj.price25)
        # adaptiveprice25 = [e/div2 for e in movingaverage(adaptiveObj.price25, window)]
        # adaptiveprice75 = [e/div2 for e in movingaverage(adaptiveObj.price75, window)]
        # aucprice25 = [e/div2 for e in movingaverage(auctioneerObj.price25, window)]
        # aucprice75 = [e/div2 for e in movingaverage(auctioneerObj.bid75, window)]
        # adaptivebidstd1 = movingaverage([(e - s)/div2 for e, s in zip(adaptiveObj.avgbiddings, adaptiveObj.bidstd)], window)
        # adaptivebidstd2 = movingaverage([(e + s)/div2 for e, s in zip(adaptiveObj.avgbiddings, adaptiveObj.bidstd)], window)
        # aucbidstd1 = movingaverage([(e - s)/div2 for e, s in zip(auctioneerObj.avgbiddings, auctioneerObj.bidstd)], window)
        # aucbidstd2 = movingaverage([(e + s) / div2 for e, s in zip(auctioneerObj.avgbiddings, auctioneerObj.bidstd)], window)
        #
        # adaptivepricestd1 = movingaverage([(e - s)/div2 for e, s in zip(adaptiveObj.avgprices, adaptiveObj.pricestd)], window)
        # adaptivepricestd2 = movingaverage([(e + s)/div2 for e, s in zip(adaptiveObj.avgprices, adaptiveObj.pricestd)], window)
        # aucpricestd1 = movingaverage([(e - s)/div2 for e, s in zip(auctioneerObj.avgprices, auctioneerObj.pricestd)], window)
        # aucpricestd2 = movingaverage([(e + s) / div2 for e, s in zip(auctioneerObj.avgprices, auctioneerObj.pricestd)], window)

        # x = range(len(adaptivebidstd1))
        # ax.fill_between(x, adaptivebid25, adaptivebid75, facecolor='green',alpha=0.5)
        # ax.fill_between(x, aucbid25, aucbid75, facecolor='yellow',alpha=0.5)
        # ax.fill_between(x, adaptivebidstd1, adaptivebidstd2, facecolor='green',alpha=0.5)
        # ax.fill_between(x, aucbidstd1, aucbidstd2, facecolor='orange',alpha=0.5)
        ax1.plot([e/div2 for e in movingaverage(adaptiveObj.avgbiddings, window)], linestyle = '--', color = 'royalblue')
        ax1.plot([e/div2 for e in movingaverage(auctioneerObj.avgbiddings, window)], linestyle = '-.', color = 'darkorange')
        # plt.legend(['w/o Auc.', 'Auc.'])
        print(j, xticklist[j])
        ax1.set_title(xticklist[j])
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if j == 0:
            ax1.set_ylabel('bids (mov. avg)')
            plt.legend(['w/o Auc.', 'w/ Auc.'], loc='best')


        # if j == 2:
        #     ax1.set_xticks([len(movingaverage(adaptiveObj.totallinks, window))/2], ['time steps'])
        #     ax2.set_xticks([len(movingaverage(adaptiveObj.totallinks, window)) / 2], ['time steps'])


        ax1.xaxis.set_visible(False)
        # ax = fig.add_axes(axes_list_10[1][j])
        plt.figure(2)
        ax2 = fig2.add_axes(axes_list_5[j])
        # ax.fill_between(x, adaptivepricestd1, adaptivepricestd2, facecolor='green',alpha=0.5)
        # ax.fill_between(x, aucpricestd1, aucpricestd2, facecolor='orange',alpha=0.5)
        # ax.fill_between(x, adaptivebidstd1, adaptivebidstd2, facecolor='green',alpha=0.5)
        # ax.fill_between(x, aucbidstd1, aucbidstd2, facecolor='orange',alpha=0.5)
        ax2.plot([e/div2 for e in movingaverage(adaptiveObj.avgprices, window)], linestyle = '--', color = 'royalblue')
        ax2.plot([e/div2 for e in movingaverage(auctioneerObj.avgprices, window)], linestyle = '-.', color = 'darkorange')
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if j == 2:
            # ax2.xaxis.set_visible(False)
            plt.xticks([len(movingaverage(adaptiveObj.totallinks, window))/2], ['time steps'])
        else:
            ax2.xaxis.set_visible(False)

        if j == 0:
            ax2.set_ylabel('prices (mov. avg)')
        # plt.legend(['w/o Auc.', 'Auc.'])
        # fig.add_axes(axes_list_15[2][j])
        # plt.plot(movingaverage(adaptiveObj.totallinks, window), linestyle = '--')
        # plt.plot(movingaverage(auctioneerObj.totallinks, window), linestyle = '-.')
        # plt.plot(movingaverage(centralObj.totallinks, window))

    plt.legend(['w/o Auc.', 'Auc.'])
    # plt.savefig('biddings_prices_links.pdf', bbox_inches='tight')
    fig1.savefig('biddings_prices_links_u.eps', format='eps', dpi=my_dpi, bbox_inches='tight')
    fig2.savefig('biddings_prices_links_l.eps', format='eps', dpi=my_dpi, bbox_inches='tight')


    # plt.show()

    # fig = plt.figure(figsize=(15, 6), dpi=my_dpi)

    for j, des in enumerate(hardcoded_designs):
        plt.figure(3)
        if j<=1:
            numPlayers = 2
        else:
            numPlayers = 3

        fops0 = fops_dict[numPlayers]['centralized']
        fops_adaptive = fops_dict[numPlayers]['adaptive']
        fops_auctioneer = fops_dict[numPlayers]['auctioneer']

        centralObj = objdict[des][fops0]
        adaptiveObj = objdict[des][fops_adaptive]
        auctioneerObj = objdict[des][fops_auctioneer]

        ax = fig3.add_axes(axes_list_5[j])

        plt.plot(movingaverage(adaptiveObj.totallinks, window), linestyle = '--', color = 'royalblue')
        plt.plot(movingaverage(auctioneerObj.totallinks, window), linestyle = '-.', color = 'darkorange')
        plt.title(xticklist[j])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if j == 0:
            plt.ylabel('shared links (mov. avg)')
            plt.legend(['w/o Auc.', 'w/ Auc.'], loc='best')

        if j == 2:
            plt.xticks([len(movingaverage(adaptiveObj.totallinks, window))/2], ['time steps'])

        else:
            ax.xaxis.set_visible(False)

        # plt.plot(movingaverage(centralObj.totallinks, window))
        # plt.legend(['w/o Auc.', 'Auc.'])
        # if j>0:
        #     ax.yaxis.set_visible(False)
        plt.figure(4)
        ax = fig4.add_axes(axes_list_5[j])
        tv_adaptive = [e/divider for e in adaptiveObj.totalcash]
        tv_auctioneer = [e/divider for e in auctioneerObj.totalcash]
        tv_centralized = [e/divider for e in centralObj.totalcash]
        # tv_adaptive = [sum(adaptiveObj.federatecash)/len(tv_adaptive) for e in tv_adaptive]
        # tv_auctioneer = [sum(auctioneerObj.federatecash)/len(tv_auctioneer) for e in tv_auctioneer]
        # tv_centralized = [sum(centralObj.federatecash)/len(tv_centralized) for e in tv_centralized]
        plt.boxplot([tv_adaptive, tv_auctioneer, tv_centralized], whis = 3)#, tv_centralized], whis = 3)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        if j == 0:
            plt.ylabel('collective value')
        plt.xticks([1,2,3],['w/o Auc.', 'w/ Auc.', '$c_f=\epsilon$'], rotation=45)

        # plt.xticks([1,2],['w/o Auc.', 'w/ Auc.'], rotation=45)


        # plt.plot(movingaverage(tv_adaptive, window))
        # plt.plot(movingaverage(tv_auctioneer, window))
        # plt.plot(movingaverage(tv_centralized, window))


    # plt.legend(['w/o Auc.', 'Auc.', ''])
    # plt.savefig('links_values.pdf', bbox_inches='tight')
    fig3.savefig('shared_links.eps', format='eps', dpi= my_dpi,bbox_inches='tight')
    fig4.savefig('collective_value_dist.eps', format='eps', dpi= my_dpi,bbox_inches='tight')
    # objlist = pickle.load(open(datapath+'Design_objectlists.p', 'rb'))




# this part plots the propagation model

# seed = 7
# random.seed(seed)
# elementlist = ['%d.%s'%(i, e) for i, e in enumerate(['SUR1', 'SUR3', 'SUR5', 'MEO1', 'MEO2', 'MEO4','MEO5', 'LEO4', 'LEO3', 'LEO1', 'GEO1', 'GEO3', 'GEO5'])]
# # elementlist = ['%d.%s'%(i, e) for i, e in enumerate(['SUR1', 'SUR3', 'SUR5', 'MEO1', 'MEO2', 'LEO4', 'LEO3', 'GEO1'])]
# # federates = [random.choice([0,1,2]) for e in elementlist]
# federates = [1, 2, 0, 1, 0, 0, 2, 0, 1, 2, 1, 2, 0]
# print(federates)
# federateDict = {e: f for e, f in zip(elementlist, federates)}
# colors = ['gold', 'lightgreen', 'red']
# colorDict = {e: colors[federateDict[e]] for e in elementlist}
# # timesteps = 50
# N = 100
# radialpoints = []
# radiusDict = {'SUR': .5, 'LEO': 1.3, 'MEO': 2.5, 'GEO': 3.5, 'ELSE': 4.}
# drawSatelliteModel(elementlist, N)
# # drawAnimativeDesignModel(elementlist)
if __name__ == '__main__':
    db = None
    dbHost = socket.gethostbyname(socket.gethostname())
    dbHost = "127.0.0.1"
    # dbHost = "155.246.119.10"
    dbName = None
    dbPort = 27017

    db = pymongo.MongoClient(dbHost, dbPort).ofs
    seed1 = 0
    seed2 = 300
    numTurns = 10000
    divider =   1000000
    div2 = 1000
    window = 120
    my_dpi = 500
    epsilon = 10
    design_id_dict = {}
    s_1 = s_3 = 0.05
    s_5 = 0.04
    s_2 = 0.05
    d_2 = 0.5
    d_3 = 0.33
    d_5 = 0.2
    w_1 = 0.85
    w_2 = 0.4
    w_3 = 0.28
    w_5 = 0.16
    fops_dict = {2: {'centralized': 'x10,-1,-1 x10,-1,-1', 'adaptive': 'x-2,-1,-1 x-2,-1,-1', 'auctioneer':'x-2,-1,1 x-2,-1,1'}, 3: {'centralized': 'x10,-1,-1 x10,-1,-1 x10,-1,-1', 'adaptive': 'x-2,-1,-1 x-2,-1,-1 x-2,-1,-1', 'auctioneer':'x-2,-1,1 x-2,-1,1 x-2,-1,1'}}


    axes_list_3 =  [[s_3, s_1, w_3, w_1], [s_3+1*d_3, s_1, w_3, w_1], [s_3+2*d_3, s_1, w_3, w_1]]
    axes_list_5 = [[s_5, s_1, w_5, w_1], [s_5+1*d_5, s_1, w_5, w_1], [s_5+2*d_5, s_1, w_5, w_1], [s_5+3*d_5, s_1, w_5, w_1], [s_5+4*d_5, s_1, w_5, w_1]]
    axes_list_10 = ([[[s_5, s_2+d_2, w_5, w_2], [s_5+1*d_5, s_2+d_2, w_5, w_2], [s_5+2*d_5, s_2+d_2, w_5, w_2], [s_5+3*d_5, s_2+d_2, w_5, w_2], [s_5+4*d_5, s_2+d_2, w_5, w_2]],
                     [[s_5, s_2, w_5, w_2],     [s_5+1*d_5, s_2, w_5, w_2],     [s_5+2*d_5, s_2, w_5, w_2],     [s_5+3*d_5, s_2, w_5, w_2],     [s_5+4*d_5, s_2, w_5, w_2]]])

    axes_list_15 = ([[[s_5, s_3+2*d_3, w_5, w_3], [s_5+1*d_5, s_3+2*d_3, w_5, w_3], [s_5+2*d_5, s_3+2*d_3, w_5, w_3], [s_5+3*d_5, s_3+2*d_3, w_5, w_3], [s_5+4*d_5, s_3+2*d_3, w_5, w_3]],
                        [[s_5, s_3+d_3, w_5, w_3],     [s_5+1*d_5, s_3+d_3, w_5, w_3],     [s_5+2*d_5, s_3+d_3, w_5, w_3],     [s_5+3*d_5, s_3+d_3, w_5, w_3],     [s_5+4*d_5, s_3+d_3, w_5, w_3]],
                            [[s_5, s_3, w_5, w_3],     [s_5+1*d_5, s_3, w_5, w_3],     [s_5+2*d_5, s_3, w_5, w_3],     [s_5+3*d_5, s_3, w_5, w_3],     [s_5+4*d_5, s_3, w_5, w_3]]])

    datapath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/'
    objdict = pickle.load(open(datapath + 'Design_objectdict.p', 'rb'))
    print("data path:", datapath)
    # saveDesignsObjPickles(hardcoded_designs, seed1, seed2, numTurns, datapath)
    # drawPricing()
    # testPickles(datapath)
    # drawPricing()
    # drawAdaptiveAuctioneer({'experiment': 'Adaptive Cost V2', 'capacity': 2, 'links': 2, 'numTurns': 10000})
    # drawTotalAdaptive({'experiment': 'Adaptive Cost V4', 'capacity': 2, 'links': 2, 'numTurns': 10000})
    # drawAdaptiveSGL({'experiment': 'Adaptive Cost V4', 'capacity': 2, 'links': 2, 'numTurns': 10000}, test = 'federate adaptive')
    # drawFederateAdaptive()

    # for i, des in enumerate(hardcoded_designs):
    #     drawGraphbyDesign(i+1, des)

# drawPricing()

# #
# drawStoragePenalty(db)
#
# drawFederateAdaptive(db)
# drawBidding()
# drawPricing()
#
# drawTotalAdaptive({'experiment': 'Adaptive Cost V2', 'capacity': 2, 'links': 2, 'numTurns': 2400})
# #
# drawAdaptiveAuctioneer({'experiment': 'Adaptive Cost V2', 'capacity': 2, 'links': 2, 'numTurns': 2400})
# #
# # drawSampleNetwork()
# #
# # drawStorageCoefficient(db)
# #
# # for i, des in enumerate(['1.GroundSta@SUR1 2.Sat@LEO2 3.Sat@MEO3 2.Sat@GEO01', '1.GroundSta@SUR1 2.Sat@LEO4 3.Sat@MEO4 2.Sat@GEO01','1.GroundSta@SUR1 2.Sat@LEO6 3.Sat@MEO5 2.Sat@GEO01', '1.GroundSta@SUR1 2.Sat@LEO2 3.Sat@MEO6 2.Sat@GEO01', '1.GroundSta@SUR1 2.Sat@LEO4 3.Sat@MEO1 2.Sat@GEO01', '1.GroundSta@SUR1 2.Sat@LEO6 3.Sat@MEO2 2.Sat@GEO01']):
# #     drawGraphbyDesign(i+1, des)