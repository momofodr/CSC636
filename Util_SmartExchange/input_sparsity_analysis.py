import numpy as np 
import sys
import os

def turn_binary(a):
    b = bin(a)[2:]

    pad = ''
    for _ in range(8-len(b)):
        pad += '0'

    return pad + b


bit_counter = [0 for _ in range(9)]
def bit_sparsity(a):
    if type(a) != 'str':
        a = turn_binary(int(a))

    cnt = 0
    for bit in a:
        if bit == '0':
            cnt += 1

    bit_counter[cnt] += 1
    return cnt

booth_counter = [0 for _ in range(5)]
def bit_sparsity_booth(a):
    if type(a) != 'str':
        a = turn_binary(int(a))

    a =  a + '0' 

    booth_list = [a[0:3], a[2:5], a[4:7], a[6:9]]

    cnt = 0
    for item in booth_list:
        if '0' not in item or '1' not in item:
            cnt += 1

    booth_counter[cnt] += 1
    return cnt



def layer_bit_sparsity(act):
    act = np.reshape(act, [-1]).astype(np.int32)
    cnt = 0
    for element in act:
        cnt += bit_sparsity(element)

    return cnt, len(act) * 8


def layer_bit_sparsity_booth(act):
    act = np.reshape(act, [-1]).astype(np.int32)
    cnt = 0
    for element in act:
        cnt += bit_sparsity_booth(element)

    return cnt, len(act) * 4


def layer_vector_sparsity(act, kernel_size, stride, padding, vector_length=8):
    act = act.astype(np.int32)

    if padding:
        act_pad = np.zeros([act.shape[0], act.shape[1], act.shape[2] + 2*padding, act.shape[3] + 2*padding])
        act_pad[:,:,padding:-padding, padding:-padding] = act
        act = act_pad

    vector_num_per_row = act.shape[2] // (vector_length * stride)

    shape3 = vector_num_per_row if vector_num_per_row * vector_length * stride == act.shape[2] else vector_num_per_row + 1

    total_vector_num = act.shape[0]*act.shape[1]*act.shape[2]*vector_num_per_row*shape3

    cnt = 0

    if kernel_size != 1:
        total_vector_num = act.shape[0] * act.shape[1] * act.shape[2] * vector_num_per_row * shape3

        for img_id in range(act.shape[0]):
            for channel in range(act.shape[1]):
                for row in range(act.shape[2]):                        
                    for n in range(vector_num_per_row):
                        lower_bound = n * 8 * stride
                        upper_bound = min((n+1) * 8 * stride + kernel_size - 1, act.shape[3])
                        vector = act[img_id, channel, row, lower_bound : upper_bound]
                        
                        if len(vector):
                            cnt += 1
                            for element in vector:
                                if element != 0:
                                    cnt -= 1
                                    break
    else:
        channel_group_num = act.shape[1] // 3
        shape1 = channel_group_num if channel_group_num * 3 == act.shape[1] else channel_group_num + 1

        total_vector_num = act.shape[0] * shape1 * act.shape[2] * vector_num_per_row * shape3

        for img_id in range(act.shape[0]):
            for channel_group in range(channel_group_num):
                for row in range(act.shape[2]):                        
                    for n in range(vector_num_per_row):
                        width_lower_bound = n * 8 * stride
                        width_upper_bound = min((n+1) * 8 * stride + kernel_size - 1, act.shape[3])
                        channel_lower_bound = channel_group * 3
                        channel_upper_bound = min((channel_group+1) * 3, act.shape[1]) 

                        vector = act[img_id, channel_lower_bound : channel_upper_bound, row, width_lower_bound : width_upper_bound]
                        vector = np.reshape(vector, [-1])

                        if len(vector):
                            cnt += 1
                            for element in vector:
                                if element != 0:
                                    cnt -= 1
                                    break

    return cnt, total_vector_num


if __name__ == '__main__':
    
    input_list = np.load(os.path.join(sys.argv[1], 'input_weight_conv.npy'), allow_pickle=True).item()['input']
    conv_info = np.load(os.path.join(sys.argv[1], 'conv_info.npy'), allow_pickle=True).item()

    bit_level = []
    bit_level_total = []

    booth = []
    booth_total = []

    vector8 = []
    vector8_total = []

    vector4 = []
    vector4_total = []

    for i, act in enumerate(input_list):
        print('dealing with layer ', i)
        cnt_sparsity, total = layer_bit_sparsity(act)
        bit_level.append(cnt_sparsity)
        bit_level_total.append(total)

        cnt_sparsity, total = layer_bit_sparsity_booth(act)
        booth.append(cnt_sparsity)
        booth_total.append(total)

        cnt_sparsity, total = layer_vector_sparsity(act, kernel_size=conv_info['RS'][i], stride=conv_info['U'][i], padding=conv_info['Padding'][i], vector_length=8)
        vector8.append(cnt_sparsity)
        vector8_total.append(total)

        cnt_sparsity, total = layer_vector_sparsity(act, kernel_size=conv_info['RS'][i], stride=conv_info['U'][i], padding=conv_info['Padding'][i], vector_length=4)
        vector4.append(cnt_sparsity)
        vector4_total.append(total)
        
    bit_counter = np.array(bit_counter)
    booth_counter = np.array(booth_counter)

    sparsity_dict = {'bit_level':bit_level, 'bit_level_total':bit_level_total, 'booth':booth, 'booth_total':booth_total, 
                     'vector8': vector8, 'vector8_total':vector8_total, 'vector4':vector4, 'vector4_total':vector4_total, 'bit_counter':bit_counter, 'booth_counter':booth_counter}   
    
    np.save(os.path.join(sys.argv[1], 'input_sparsity_info.npy'), sparsity_dict)











