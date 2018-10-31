import random 
a = random.sample(range(1,100), 14)
b = random.sample(range(1,100), 20)
result = [i for i in set(a) if i in b]
print(result)
