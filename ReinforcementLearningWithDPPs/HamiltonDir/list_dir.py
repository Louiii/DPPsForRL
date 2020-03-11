import os
import numpy as np

entries = os.listdir('data/')

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

nums = [int(s[1:-5]) for s in entries if s[1:-5].isdigit()]

nums = set(sorted(nums))

required = set(list(range(1000)))

complement = required - nums

print(complement)
print(len(complement))

multiples = {}
for c in list(complement):
    key = c%50
    if key in multiples:
        multiples[key].append(c)
    else:
        multiples[key] = [c]
print()
print(multiples)

multip = {k:len(lst) for k,lst in multiples.items()}
print('\n\n')
print(multip)
print()
numas1 = [np.array(multiples[i])-i for i in range(28)]

numas2 = [(list(np.array(multiples[i])-i), str(list(np.array(multiples[i])-i))=='[100, 150, 200]') for i in range(28, 50)]
n2a = [a for (a, b) in numas2 if b]
n2b = [a for (a, b) in numas2 if not b]
print(numas1)
print('\n')
print(n2a)
print(len(n2a))
print('\n')
print(n2b)
print(len(n2b))

