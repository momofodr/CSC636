import numpy as np 
import sys
import os


layers = np.load(sys.argv[1], allow_pickle=True)

Bs = {}
Ces = {}

for i, lname in enumerate(layers):
    layer = layers[lname].item()
    B = {}
    Ce = {}
    for key, kernel in layer.items():
        if 'k' in key or 'r' in key:
            B[key] = kernel['Bs']
            Ce[key] = kernel['Ces']
    Bs['l'+str(i+1)] = B
    Ces['l'+str(i+1)] = Ce

np.save('B.npy', Bs)
np.save('Ce.npy', Ces)










