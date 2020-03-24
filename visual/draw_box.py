import numpy as np
import sys
import os
import matplotlib.pyplot as plt


fnames = os.listdir(sys.argv[1])

model2sparsity_bit = {}
model2sparsity_booth = {}

for fname in fnames:
    bit_counter = np.load(os.path.join(sys.argv[1], fname), allow_pickle=True).item()['bit_counter']
    booth_counter = np.load(os.path.join(sys.argv[1], fname), allow_pickle=True).item()['booth_counter']
    model = fname.split('.')[0].split('_')[-1]
    if model == 'MobileNetV1':
        model = 'MBV1'
    elif model == 'MobileNetV2':
        model = 'MBV2'
    model2sparsity_bit[model] = bit_counter
    model2sparsity_booth[model] = booth_counter


labels_bit = list(model2sparsity_bit.keys())
cnt_sparsity_bit = list(model2sparsity_bit.values())

data_bit = []
for item in cnt_sparsity_bit:
    data_bit_item = []
    for i in range(len(item)):
        data_bit_item.extend([8-i for _ in range(int(item[i]))])
    data_bit.append(data_bit_item)


labels_booth = list(model2sparsity_booth.keys())
cnt_sparsity_booth = list(model2sparsity_booth.values())

data_booth = []
for item in cnt_sparsity_booth:
    data_booth_item = []
    for i in range(len(item)):
        data_booth_item.extend([4-i for _ in range(int(item[i]))])
    data_booth.append(data_booth_item)


font_big = 19
font_mid = 16
font_small_y = 12
font_small_x = 9


fig, ax = plt.subplots(1, 2, figsize=(10,8))
plt.subplots_adjust(wspace=0.3, hspace=0.35)

ax[0].boxplot(data_bit, labels=labels_bit)
ax[0].set_title('Bit-Level Sparsity Distribution', fontsize=font_big, fontweight='bold')
ax[0].set_xlabel('Model', fontsize=font_mid, fontweight='bold')
ax[0].set_ylabel('#1 in 8-bit Activation', fontsize=font_mid, fontweight='bold')
ax[0].xaxis.set_tick_params(labelsize=font_small_x)
ax[0].yaxis.set_tick_params(labelsize=font_small_y)

ax[1].boxplot(data_booth, labels=labels_booth)
ax[1].set_title('Booth Sparsity Distribution', fontsize=font_big, fontweight='bold')
ax[1].set_xlabel('Model', fontsize=font_mid, fontweight='bold')
ax[1].set_ylabel('#1 in 8-bit Activation', fontsize=font_mid, fontweight='bold')
ax[1].xaxis.set_tick_params(labelsize=font_small_x)
ax[1].yaxis.set_tick_params(labelsize=font_small_y)

plt.savefig('sparsity_box.pdf', bbox_inches='tight')



 