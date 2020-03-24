import numpy as np 
import sys
import os


def compute_overhead(encoding, bit=4):
    cnt_overhead = 0
    cnt_zero = 0
    for item in encoding:
        if item or cnt_zero == 2**bit:
            cnt_overhead += 1
            cnt_zero = 0
        else:
            cnt_zero += 1

    return bit * cnt_overhead



def row_sparsity(weight):
    cnt_sparsity_list = []
    cnt_total_list = []
    overhead_naive_list = []
    overhead_ex4_list = []
    overhead_ex2_list = []

    for outchannel in range(weight.shape[0]):
        cnt = 0
        encoding = []

        for row in range(weight.shape[2]):
            for inchannel in range(weight.shape[1]):
                vector = weight[outchannel, inchannel, row, :]
                
                flag_sparse = 0
                cnt += 1
                for element in vector:
                    if element != 0:
                        flag_sparse = 1
                        cnt -= 1
                        break

                encoding.append(flag_sparse)

        cnt_sparsity_list.append(cnt)
        overhead_naive_list.append(weight.shape[1]*weight.shape[2])
        overhead_ex4_list.append(compute_overhead(encoding, bit=4))
        overhead_ex2_list.append(compute_overhead(encoding, bit=2))

    return np.sum(cnt_sparsity_list), np.sum(overhead_naive_list), np.sum(overhead_ex4_list), np.sum(overhead_ex2_list)



if __name__ == '__main__':
    
    weight_list = np.load(os.path.join(sys.argv[1], 'input_weight_conv.npy'), allow_pickle=True).item()['weight']
    conv_info = np.load(os.path.join(sys.argv[1], 'conv_info.npy'), allow_pickle=True).item()

    cnt_sparsity_list = []
    overhead_naive_list = []
    overhead_ex4_list = []
    overhead_ex2_list = []

    for i, weight in enumerate(weight_list):
        print('dealing with layer ', i)
        cnt_sparsity, overhead_naive, overhead_ex4, overhead_ex2 = row_sparsity(weight) 

        cnt_sparsity_list.append(cnt_sparsity)
        overhead_naive_list.append(overhead_naive)
        overhead_ex4_list.append(overhead_ex4)
        overhead_ex2_list.append(overhead_ex2)

    stat_total = {'cnt_sparsity':np.sum(cnt_sparsity_list), 'overhead_naive': np.sum(overhead_naive_list), 'overhead_ex4':np.sum(overhead_ex4_list), 'overhead_ex2':np.sum(overhead_ex2_list)}
        
    sparsity_dict = {'cnt_sparsity':cnt_sparsity_list, 'overhead_naive': overhead_naive_list, 'overhead_ex4':overhead_ex4_list, 'overhead_ex2':overhead_ex2_list, 'stat_total': stat_total}                 
    
    np.save(os.path.join(sys.argv[1], 'weight_sparsity_info.npy'), sparsity_dict)











