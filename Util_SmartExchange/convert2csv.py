import numpy as np 
import sys
import os
import csv

def npy2csv(fname):
    info_dict = np.load(fname, allow_pickle=True).item()

    csv_file = os.path.join(os.path.dirname(fname), os.path.basename(fname).split('.')[0]+'.csv') 
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        keys = list(info_dict.keys())
        for i, key in enumerate(keys):
            if 'stat_total' in key:
                keys.pop(i)

        writer.writerow(keys)

        for i in range(len(info_dict[keys[0]])):
            row = []
            for key in keys:
                row.append(info_dict[key][i])
            writer.writerow(row)



if __name__ == '__main__':
    npy2csv(os.path.join(sys.argv[1], 'input_sparsity_info.npy'))
    npy2csv(os.path.join(sys.argv[1], 'weight_sparsity_info.npy'))












