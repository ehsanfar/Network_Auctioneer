import numpy 
import random 

ID = 0 
r = 1.077
p = 0.5 
survive = 0.5

peoplelist = []

class GGGParent(): 
	def __init__(self, ID, gender, obj1 = None, obj2 = None): 
		self.ID = ID
		self.gender = gender
		if not obj1 and not obj2: 
			self.source = set([ID])
		else:
			self.source = obj1.source.union(obj2.source)
			# print("combine:", obj1.source, obj2.source)
	
def createNewGen(oldgentlist): 
	women = [e for e in oldgentlist if e == 1]
	men = [e for e in oldgentlist if e == 0]
	p = 0.5 

initiallist = []
N = 1000
for i in range(N): 
	gender = 0 if random.random()>0.5 else 1 
	initiallist.append(GGGParent(ID, gender))
	ID += 1 

GN = 40
oldGen = initiallist
for j in range(GN): 
	newGen = []
	women = [g for g in oldGen if g.gender == 0]
	men = [g for g in oldGen if g.gender == 1]
	random.shuffle(men)
	random.shuffle(women)
	while len(newGen) <= r**(j+1) * N:
		for m, w in zip(men, women): 
			if random.random()<survive:
				gender = 0 if random.random()>0.5 else 1 
				newperson = GGGParent(ID, gender, m, w)
				newGen.append(newperson)
				ID += 1 
	oldGen = newGen[:]
	print(len(newGen))

for person in newGen:
	print(person.ID,len(person.source))

print(len(newGen))
	
