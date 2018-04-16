# import matplotlib.pyplot as plt
from .globalv import * 
import numpy as np
import networkx as nx
import re
import math
from collections import Counter, defaultdict
# from matplotlib import gridspec
import hashlib
import json
import pickle
from .result import QResult, DesignClass
import os

def pickTask(task, time):
    element = task.element
    task.lastelement = element
    element.size += task.size
    task.init = time
    task.expiration = time + 5

def transTask(task, link, cost, solutionObj):
    # link.source.size -= task.size
    # link.destin.size += task.size
    task.lastelement = link.destin
    taskfedname = task.element.owner.name
    solutionObj.addValue(taskfedname, -1*max(cost, epsilon)) 
    linkfedname = link.owner.name
    solutionObj.addValue(linkfedname, max(cost, epsilon) - epsilon)
    # solutionObj.fedValDict[linkfedname] += (cost - epsilon)
    # task.element.owner.cash -= cost
    # link.owner.cash += cost - epsilon

def resolveTask(task, value, solutionObj):
    taskfedname = task.element.owner.name
    solutionObj.addValue(taskfedname, value) 
    # solutionObj.fedValDict[taskfedname] += value
    # task.element.owner.cash += value
    task.element.size -= task.size

def checkEqual2(iterator):
   return len(set(iterator)) <= 1

def checkequallists(l1, l2): 
    if len(l1) == len(l2): 
        if all([a == b for a,b in zip(l1, l2)]): 
            return True
    
    return False

def findbestxy(N):
    if N % 2 != 0:
        N += 1
    temp = int(N ** 0.5)
    while N % temp != 0:
        temp -= 1

    return (temp, N // temp)

def convertPath2Edge(pathlist):
    tuplist = []
    for i in range(len(pathlist) - 1):
        tuplist.append((pathlist[i], pathlist[i + 1]))

    return tuplist

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


def convertPath2StaticPath(path):
    temppath = [e[:-2] for e in path.nodelist]
    ends = [e[-1] for e in path.nodelist]
    seen = set([])
    seen_add = seen.add
    staticpath = [e for e in temppath if not (e in seen or seen_add(e))]
    # print "convert path 2 static path:", path, staticpath
    deltatime = path.deltatime
    assert len(set(ends[deltatime:])) == 1
    return (staticpath, deltatime)


def bfs_paths(G, source, destination):
    queue = [(source, [source])]
    while queue:
        v, path = queue.pop(0)
        for next in set(G.neighbors(v)) - set(path):
            if next == destination:
                yield path + [next]
            else:
                queue.append((next, path + [next]))

def findAllPaths(G, sources, destinations):
    allpathes = []
    for s in sources:
        for d in destinations:
            allpathes.extend(bfs_paths(G, s, d))

    return allpathes

# class Path():
#     def __init__(self, l):
#         self.linklist = l
def findClosestIndex(value, valulist):
    abslist = [abs(v-value) for v in valulist]
    return abslist.index(min(abslist))

def addDict2Dict(dict1, dict2):
    dict3 = dict1.copy()
    for d, c in dict2.items():
        dict3[d] += c
    return dict3

def createHash(experiment, numfederates, numElements, sharelinkcost, uselinkcost, seed):
    m = hashlib.md5()
    resultsstr = "%s %s %s %s %s %s" % (experiment, str(numfederates), str(numElements).zfill(2), 
        str(sharelinkcost).zfill(4), str(uselinkcost).zfill(4), str(seed).zfill(3))
    print(resultsstr)
    ustr = resultsstr.encode('utf-16')
    m.update(ustr)
    # print(resultsstr, m.hexdigest())
    return str(m.hexdigest())

# def avgSeeds()


