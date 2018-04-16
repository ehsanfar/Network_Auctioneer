from gurobipy import Model, LinExpr, GRB, GurobiError
from .generalFunctions import *
from .globalv import * 
from .classes import *
    
def costfunction(f, l):
    global epsilon
    f2 = l.destin.owner
    if f.name == f2.name:
        return epsilon
    else:
        return f2.sharelinkcost

def costfunction(f, l, edgePriceDict):
    global epsilon
    f2 = l.destin.owner
    if f.name == f2.name:
        return epsilon
    else:
        return max(epsilon, edgePriceDict[(l.source.name, l.destin.name)])
# def optimizewithPrice(elements, linklist, destinations, storedtasks, newtasks, time, federates): 

def optimizeMILP(elements, linklist, destinations, storedtasks, newtasks, time, federates, edgePriceDict, solutionObj):
    global storagepenalty, epsilon, linkcapacity, elementcapacity
    # print("MILP solution bid dict:", solutionObj.fedBidDict)
    tasklist = storedtasks + newtasks
    lp = Model('LP')
    steps =  10
    timesteps = range(time, time + steps)
    trans = [] # trans[t][i][l] transfer task i from link l at time t
    store = [] # store[i][j] store task i
    pick = []   # pick[i] if source i picks up the task
    resolve = []

    J = LinExpr()

    for i, task in enumerate(tasklist):
        store.insert(i, lp.addVar(vtype=GRB.BINARY))
        J.add(store[i], -1* storagepenalty)
        r = LinExpr()
        r.add(store[i], 1)
        lp.addConstr(r <= 1)
        # lp.addConstr(r == 0)

    for i, task in enumerate(tasklist):
        pick.append(lp.addVar(vtype=GRB.BINARY))
        J.add(pick[i], -1)
        element = task.element
        r = LinExpr()
        r.add(pick[i], 1)
        if task.init < time:
            lp.addConstr(r == 1)
        else:
            lp.addConstr(r <= 1)

    for i, t in enumerate(timesteps):
        trans.insert(i, [])
        resolve.insert(i, [])
        for k, task in enumerate(tasklist):
            trans[i].insert(k, [])
            resolve[i].insert(k, [])
            for j, e in enumerate(elements):
                resolve[i][k].insert(j, lp.addVar(vtype=GRB.BINARY))
                if e.name in destinations:
                    J.add(resolve[i][k][j], task.maxvalue)
                else:
                    J.add(resolve[i][k][j], task.penalty)

            if i == 0 and (task.expiration <= time):
                r = LinExpr()
                element = task.element
                j, e = next(((a, b) for a, b in enumerate(elements) if b.name == element.name))
                r.add(resolve[i][k][j], 1)
                lp.addConstr(r == 1)

            for l, link in enumerate(linklist):
                trans[i][k].insert(l, lp.addVar(vtype=GRB.BINARY))
                J.add(trans[i][k][l], -1*epsilon)

                r = LinExpr()
                r.add(trans[i][k][l], 1)
                lp.addConstr(r <= (1 if (task.size <= (link.capacity - link.size)
                                         and link.source.name not in destinations) else 0))

                r.add(pick[k], -1)
                lp.addConstr(r <= 0)

                r = LinExpr()
                r.add(sum(trans[i][k]))
                lp.addConstr(r <= 1)
                d = link.destin
                j, e = next(((a, b) for a, b in enumerate(elements) if b.name == d.name))
                r = LinExpr()
                r.add(resolve[i][k][j], 1)
                lp.addConstr(r <= (1 if (d.name in destinations) else 0))

    for i, t in enumerate(timesteps):
        for k, task in enumerate(tasklist):
            for j, element in enumerate(elements):
                inlinks = [(l, li) for l, li in enumerate(linklist) if li.destin.name == element.name]
                outlinks = [(l, li) for l, li in enumerate(linklist) if li.source.name == element.name]
                if i == 0 and element.name == task.element.name:
                    r = LinExpr()
                    for l, li in outlinks:
                        r.add(trans[i][k][l], -1)

                    r.add(resolve[i][k][j], -1)
                    r.add(store[k], -1)
                    r.add(pick[k], 1)
                    lp.addConstr(r == 0)
                elif element.name in destinations:
                    r = LinExpr()
                    # r2 = LinExpr()
                    for l, li in inlinks:
                        r.add(trans[i][k][l], 1)

                    r.add(resolve[i][k][j], -1)
                    lp.addConstr(r == 0)

                else:
                    r = LinExpr()
                    # r2 = LinExpr()
                    for l, li in inlinks:
                        r.add(trans[i][k][l], 1)

                    r.add(resolve[i][k][j], -1)
                    if i< len(timesteps) - 1:
                        for l, li in outlinks:
                            r.add(trans[i+1][k][l], -1)

                    lp.addConstr(r == 0)

    #
    for k, task in enumerate(tasklist):
        r = LinExpr()
        r.add(pick[k], -1)
        r.add(store[k], 1)
        for j, element in enumerate(elements):
            for i, t in enumerate(timesteps):
                r.add(resolve[i][k][j], 1)
        lp.addConstr(r == 0)


    for l, li in enumerate(linklist):
        r = LinExpr()
        for k in range(len(tasklist)):
            for i in range(len(timesteps)):
                r.add(trans[i][k][l])

        lp.addConstr(r <= linkcapacity)

    for j, e in enumerate(elements):
        r = LinExpr()
        for k, task in enumerate([t for t in tasklist if e.name == task.element.name]):
            r.add(pick[k], 1)
            for i in range(len(timesteps)):
                for v in range(len(elements)):
                    r.add(resolve[i][k][v], -1)

        lp.addConstr(r <=  elementcapacity)
    # for i in range(len(timesteps)):
    #     rl = [LinExpr() for e in elements]
    #     for k, task in enumerate(tasklist):
    #         element = task.element
    #         j, e = next(((a, b) for a, b in enumerate(elements) if b.name == element.name))
    #         rl[j].add(store[k], 1)
    #         rl[j].add(resolve[0][k][j], -1)
    #
    #     for r in rl:
    #         lp.addConstr(r <= elementcapacity)
    for k, task in enumerate(tasklist):
        r = LinExpr()
        fedtask = task.element.owner
        for i in range(len(timesteps)):
            for l, li in enumerate(linklist):
                r.add(trans[i][k][l], -1*costfunction(fedtask, li, edgePriceDict))

        # r.add(task.getValue(time), 1)
        # r.add(fedtask.uselinkcost, 1)
        r.add(min(task.getValue(time), solutionObj.fedBidDict[fedtask.name][1]), 1)
        lp.addConstr(r >= 0)


    lp.setObjective(J, GRB.MAXIMIZE)
    lp.setParam('OutputFlag', False)
    lp.optimize()
    for i, task in enumerate(newtasks):
        if pick[i].x>0.5:
            pickTask(task, time)

    edges = []
    sourceEdgeDict = defaultdict(list)
    for i, t in enumerate(timesteps):
        for k, task in enumerate(tasklist):
            for l, link in enumerate(linklist):
                if trans[i][k][l].x>0.5:
                    edges.append((link.source.name, link.destin.name))
                    sourceEdgeDict[task.element.name].append((link.source.name, link.destin.name))
                    if task.element.owner == link.owner:
                        transTask(task, link, epsilon, solutionObj)
                    else:
                        transTask(task, link, costfunction(task.element.owner, link, edgePriceDict), solutionObj)

            for j, e in enumerate(elements):
                if resolve[i][k][j].x>0.5:
                    # if task.expiration <= time:
                    #     resolveTask(task, task.penalty)
                    # else:
                    #     resolveTask(task, task.value)
                    resolveTask(task, task.getValue(time), solutionObj)

    for k, task in enumerate(tasklist):
        net = 0
        fedtask = task.element.owner
        for i in range(len(timesteps)):
            for l, li in enumerate(linklist):
                net -= trans[i][k][l].x * costfunction(fedtask, li, edgePriceDict)

        net += task.getValue(time)

    storedtasks = []
    for k, task in enumerate(tasklist):
        if (pick[k].x and store[k].x) and not any([resolve[i][k][j].x for i, j in product(range(len(timesteps)), range(len(elements)))]):
            storedtasks.append(task)
    
    # solutionObj.edgelist.extend(edges)
    solutionObj.sourceEdgeDict = sourceEdgeDict
    return solutionObj

    