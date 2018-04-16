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

from .globalv import *
from collections import defaultdict 
import hashlib

class Federate():
    def __init__(self, name, cash):
        global storagepenalty
        self.name = name
        self.cash = cash
        self.storagepenalty = storagepenalty 

class Link():
    def __init__(self, source, destin, capacity, size, owner):
        self.source = source
        self.destin = destin
        self.capacity = capacity
        self.size = size
        self.owner = owner

class Task():
    def __init__(self, id, element, lastelement, size, value, expiration, init, active, penalty):
        self.id = id
        self.element = element
        self.lastelement = lastelement
        self.size = size
        self.expiration =expiration
        self.init = init
        self.active = active
        self.penalty = penalty
        self.maxvalue = value

    def getValue(self, time):
        """
        Gets the current value of this contract.
        @return: L{float}
        """
        # print time, self.initTime
        duration = self.expiration - self.init + 1
        self.elapsedTime = time - self.init
        value = self.maxvalue if self.elapsedTime <= duration else self.penalty if self.elapsedTime > self.expiration \
            else self.maxvalue * (1. - self.elapsedTime) / (2. * self.expiration)
        return value

class Element():
    def __init__(self, name, capacity, size, owner):
        self.name = name
        self.capacity = capacity
        self.size = size
        self.owner = owner

class NetTop(): 
   def __init__(self, elements, edges, federates, sources, destinations):
      tempelfeddict = {e:f for e,f in zip(elements, federates)}
      self.elements = sorted(elements, key = lambda x: x)
      self.edges = sorted(edges)
      self.federates = [tempelfeddict[e] for e in self.elements]
      self.sources = sorted(sources)
      self.destinations = sorted(destinations)
      self.costValueDict = defaultdict(float)
      self.auctionValueDict = {}
      self.auctionscore = 0 
      self.hashid = self.createHash()
    
   def __hash__(self):
     return self.hashid

   def createHash(self): 
     m = hashlib.md5()
     resultsstr = ''.join(self.elements) + ''.join([''.join(e) for e in self.edges]) + ''.join(self.federates) + ''.join(self.destinations) + ''.join(self.sources)
     ustr = resultsstr.encode('utf-16')
     m.update(ustr)
     return  str(m.hexdigest())
     
class Path(): 
    def __init__(self, elementlist): 
        self.elementlist = elementlist
        self.linklist = getLinks()
    
    def getLinks(self): 
        templinklist = []
        for e1, e2 in zip(self.elementlist[:-1], self.elementlist[1:]): 
            templinklist.append((e1, e2))
        return templinklist

class Bundle(): 
   def __init__(self, pathlist): 
      self.pathlist = pathlist
   
   def addpath(path): 
      self.pathlist.append(path)
   
class MILPSolution(): 
   def __init__(self, nethashid, time, fedBidDict, edgelist): 
      self.time = time
      self.nettophashid = nethashid
      self.edgelist = edgelist
      self.fedBidDict = fedBidDict
      # self.totalvalue = sum(self.fedValDict.values())
      self.independentValues = []
      self.independentPathlist = []
      self.centralizedValues = []
      self.centralizedPathlist = []
      self.federatedPathlist = []
      self.federatedValues = []
      self.centralizedMILPValues = []
      self.centralizedMILPPrices = []
      self.centralizedSLSQPValues = []
      self.centralizedSLSQPPrices = []
      self.federatedMILPValues = []
      self.federatedMILPPrices = []
      self.federatedSLSQPValues = []
      self.federatedSLSQPPrices = [] 
      # self.Bundle = None
      # self.sourceEdgeDict = defaultdict(list)
      # self.edgePriceDict = defaultdict(float)
      self.hashid = self.createHash()
   
   def addValue(self, fname, value): 
      self.fedValDict[fname] += value
      self.totalvalue = sum(self.fedValDict.values())
   
   def addEdge(self, edge): 
      self.edges.append(edge)
   
   def addPath(self, edgelist):
      self.edges.extend(edgelist)
   
   def __hash__(self):
     return self.hashid

   def createHash(self): 
     # m = hashlib.md5()
     pricelist = [e for f, t in sorted(self.fedBidDict.items()) for e in t]
     resultsstr = self.nettophashid + ''.join(list([str(int(e)).zfill(4) for e in pricelist]))
     # ustr = resultsstr.encode('utf-16')
     # m.update(ustr)
     # return  str(m.hexdigest())
     return resultsstr


class LiteSolution(): 
   def __init__(self, fedBidDict, nettopObj): 
      nf = len(set(nettopObj.federates))
      self.fedValDict = {'f%d'%i: 0 for i in range(nf)}
      self.totalvalue = 0
      self.sourceEdgeDict = []
      self.nettophashid = nettopObj.hashid
      self.fedBidDict = fedBidDict
      self.hashid = self.createHash()

   
   def addValue(self, fname, value): 
      self.fedValDict[fname] += value
      self.totalvalue = sum(self.fedValDict.values())
      
   def reset(self): 
      self.fedValDict = {'f%d'%i: 0 for i in range(len(self.fedValDict))}
      self.totalvalue = 0

   def __hash__(self):
     return self.hashid

   def createHash(self): 
     # pricelist = [e for t in sorted(self.fedBidDict.values()) for e in t]
     pricelist = [e for f, t in sorted(self.fedBidDict.items()) for e in t]
     hashid = int(''.join([str(e//10).zfill(3) for e in pricelist]))
     return hashid
      
class Constraint1():
   def __init__(self, fedname, pathedgelist, elfedDict, fedvalue, taskvalue = 1000):
      self.name = fedname
      self.fedvalue = fedvalue
      self.pathedgelist = pathedgelist 
      self.elfedDict = elfedDict
      self.taskvalue = taskvalue
      self.fedpathlist = [elfedDict[edgelist[0][0]] for edgelist in pathedgelist if elfedDict[edgelist[0][0]] == fedname]
      self.fedtaskvalue = taskvalue * len([elfedDict[edgelist[0][0]] for edgelist in pathedgelist if elfedDict[edgelist[0][0]] == fedname]) - epsilon * sum([len(p) for p in self.fedpathlist])

   def __call__(self, pricelist):
      priceDict = {'f%d'%i: price for i, price in enumerate(pricelist)}
      linkcostDict = defaultdict(int)
      linkvalueDict = defaultdict(int)
      
      for edgelist in self.pathedgelist:
         sourcefed = self.elfedDict[edgelist[0][0]]
         for e in edgelist: 
            if self.elfedDict[e[1]] != sourcefed:
               linkcostDict[sourcefed] += priceDict[self.elfedDict[e[1]]]
               linkvalueDict[self.elfedDict[e[1]]] += priceDict[self.elfedDict[e[1]]] - epsilon
            else: 
               linkcostDict[sourcefed] += epsilon
            
               
      fedpathlist = [self.elfedDict[edgelist[0][0]] for edgelist in self.pathedgelist if self.elfedDict[edgelist[0][0]] == self.name]
      fedtaskvalue = 0 
      for edgelist in self.pathedgelist: 
         if self.elfedDict[edgelist[0][0]] == self.name:
            fedtaskvalue += self.taskvalue
         # self.taskvalue * len([self.elfedDict[edgelist[0][0]] for edgelist in self.pathedgelist if self.elfedDict[edgelist[0][0]] == self.name]) 
      return fedtaskvalue - linkcostDict[self.name] + linkvalueDict[self.name] - self.fedvalue
      # priceDict = {'f%d'%i: price for i, price in enumerate(pricelist)}
      # modifiedlist = []
      # for edgelist in self.pathedgelist: 
      #    # edgelist = list(zip(path, path[1:]))
      #    pathfed = self.elfedDict[edgelist[0][0]]
      #    pathcostlist = [priceDict[self.elfedDict[e[1]]] if self.elfedDict[e[1]] != pathfed else epsilon for e in edgelist]
      #    pathcost = sum(pathcostlist)
      #    if pathcost <= self.taskvalue: 
      #       modifiedlist.append(edgelist)
            
      # fedpathlist = [el for el in modifiedlist if self.elfedDict[el[0][0]] == self.name]            
      # fedtaskvalue = self.taskvalue * len(fedpathlist) - epsilon * sum([len(p) for p in fedpathlist])
      # outedgelist = [e for e in [e for l in modifiedlist for e in l] if self.elfedDict[e[1]] != self.name]
      # tempotherpaths =  [el for el in modifiedlist if self.elfedDict[el[0][0]] != self.name]
      # inedgelist =  [e for e in [e for l in tempotherpaths for e in l] if self.elfedDict[e[1]] == self.name]
      
      # linkcost = sum([priceDict[self.elfedDict[e[1]]] for e in outedgelist])
      # linkvalue = sum([priceDict[self.name] for e in inedgelist])
      # return fedtaskvalue + linkvalue - linkcost - self.fedvalue


class Constraint2():
   def __init__(self, edgelist, elfedDict, taskvalue = 1000):
      self.edgelist = edgelist #list(zip(path, path[1:]))
      self.taskvalue = taskvalue
      self.elfedDict = elfedDict
      self.sourcefederate = elfedDict[edgelist[0][0]]
      
   def __call__(self, pricelist):
      priceDict = {'f%d'%i: price for i, price in enumerate(pricelist)}
      # pathcost = sum([c for c, f in zip(self.path.linkcostlist, self.path.linkfederatelist) if f is self.path.elementOwner.federateOwner])
      pathcost = 0.
      for (e1, e2) in self.edgelist:
         linkfederate = self.elfedDict[e2]
         price = priceDict[self.elfedDict[e2]] if linkfederate != self.sourcefederate else epsilon
         pathcost += price

      return self.taskvalue - pathcost

class Constraint3():
   def __init__(self, pathedgelist, elfedDict, taskvalue = 1000):
      self.pathedgelist = pathedgelist
      self.taskvalue = taskvalue
      self.elfedDict = elfedDict
      
   def __call__(self, pricelist):
      priceDict = {'f%d'%i: price for i, price in enumerate(pricelist)}
      n = 0 
      for edgelist in self.pathedgelist: 
         # edgelist = list(zip(path, path[1:]))
         sourcefederate = self.elfedDict[edgelist[0][0]]
         pathcostlist = [priceDict[self.elfedDict[e[1]]] if self.elfedDict[e[1]] != sourcefederate else epsilon for e in edgelist]
         pathcost = sum(pathcostlist)
         if pathcost <= self.taskvalue: 
            n += 1 
      
      return n - len(self.pathedgelist)
               
class Objective():
   def __init__(self, pathedgelist, elfedDict, taskvalue = 1000, alpha = 1):
      self.pathedgelist = pathedgelist
      self.taskvalue = taskvalue
      self.elfedDict = elfedDict
      self.alpha = alpha

   def __call__(self, pricelist):
      priceDict = {'f%d'%i: price for i, price in enumerate(pricelist)}
      modifiedlist = []
      totalrevenue = 0
      totalpathcost = 0 
      for edgelist in self.pathedgelist: 
         # edgelist = list(zip(path, path[1:]))
         sourcefederate = self.elfedDict[edgelist[0][0]]
         pathcostlist = [priceDict[self.elfedDict[e[1]]] if self.elfedDict[e[1]] != sourcefederate else epsilon for e in edgelist]
         pathcost = sum(pathcostlist)
         if pathcost <= self.taskvalue: 
            modifiedlist.append(edgelist)
            totalrevenue += self.taskvalue - epsilon * len(pathcostlist)
            totalpathcost += pathcost
      
      return  -1 * (self.alpha * totalrevenue + (1-self.alpha) * totalpathcost)  
       
        