#make random lists and check overlapping elements 
import random

a = list(random.sample(range(1, 200), 10))
b = list(random.sample(range(1, 200), 10))
c = []

for num in a: 
	if num in b:
		if num not in c: 
			c.append(num)
print(a)
print(b)
print(c)

'''
random.sample(population, k)
Return a k length list of unique elements chosen from population sequence
Random sampling without replacement 

random.randint(a, b) - random integers such that a <= x <= b 

xrange(inclusive start, exclusive end)

random() -- generate random float 0 <= x < 1.0 

uniform(2.5, 10.0) - random float 2.5 <= x < 10.0 

randrange(start, end, 2) - even integers from start (inclusive) to end (exclusive)
'''