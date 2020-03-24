import numpy as np 
import sys
import os

dirs = os.listdir(sys.argv[1])

fnames = [os.path.join(sys.argv[1], dir, 'input_sparsity_info.npy') for dir in dirs]

for fname in fnames:
    vector4 = np.load(fname, allow_pickle=True).item()['vector4']
    vector4_total = np.load(fname, allow_pickle=True).item()['vector4_total']
    vector8 = np.load(fname, allow_pickle=True).item()['vector8']
    vector8_total = np.load(fname, allow_pickle=True).item()['vector8_total']

    print(os.path.dirname(fname))
    print('vector4')
    for i in range(len(vector4)):
        print(vector4[i]/vector4_total[i])

    print('vector8')
    for i in range(len(vector8)):
        print(vector8[i]/vector8_total[i])

    input()











