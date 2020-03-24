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


# tick_labels = list(model2sparsity_bit.keys())
tick_labels = ['VGG11','ResNet50','MBV2','VGG19','ResNet164','DeepLabV3']

cnt_sparsity_bit_orig = []
cnt_sparsity_bit_orig.append(model2sparsity_bit['VGG11'])
cnt_sparsity_bit_orig.append(model2sparsity_bit['ThiNet'])
cnt_sparsity_bit_orig.append(model2sparsity_bit['MBV2'])
cnt_sparsity_bit_orig.append(model2sparsity_bit['VGG19'])
cnt_sparsity_bit_orig.append(model2sparsity_bit['ResNet164'])
cnt_sparsity_bit_orig.append(model2sparsity_bit['DeepLabV3'])

# cnt_sparsity_bit_orig = list(model2sparsity_bit.values())
cnt_sparsity_bit = []
for item in cnt_sparsity_bit_orig:
    cnt_sparsity_bit.append(np.flip(item))

cnt_sparsity_bit = np.array(cnt_sparsity_bit)
cnt_sparsity_bit = cnt_sparsity_bit / np.sum(cnt_sparsity_bit, axis=1).reshape((-1,1)) * 100
data_bit = np.transpose(cnt_sparsity_bit)

cnt_sparsity_booth_orig = []
cnt_sparsity_booth_orig.append(model2sparsity_booth['VGG11'])
cnt_sparsity_booth_orig.append(model2sparsity_booth['ThiNet'])
cnt_sparsity_booth_orig.append(model2sparsity_booth['MBV2'])
cnt_sparsity_booth_orig.append(model2sparsity_booth['VGG19'])
cnt_sparsity_booth_orig.append(model2sparsity_booth['ResNet164'])
cnt_sparsity_booth_orig.append(model2sparsity_booth['DeepLabV3'])


# cnt_sparsity_booth_orig = list(model2sparsity_booth.values())
cnt_sparsity_booth = []
for item in cnt_sparsity_booth_orig:
    cnt_sparsity_booth.append(np.flip(item))

cnt_sparsity_booth = np.array(cnt_sparsity_booth)
cnt_sparsity_booth = cnt_sparsity_booth / np.sum(cnt_sparsity_booth, axis=1).reshape((-1,1)) * 100
data_booth = np.transpose(cnt_sparsity_booth)


font_big = 25
font_mid = 20
font_small_y = 20
font_small_x = 20
font_legend = 13

bar_width = 0.14

axis_width = 3


fig, ax = plt.subplots(1, 2, figsize=(20,9))
plt.subplots_adjust(wspace=0.15, hspace=0.35)

x = np.arange(len(tick_labels))

for i in range(len(data_bit)):
    ax[0].bar(x + i*bar_width, data_bit[i], width=bar_width, label='#1 = '+str(i))
    if i < 7:
        for a, b in zip(x, data_bit[i]):
            ax[0].text(a + i*bar_width + 0.01, b+0.3, '%.1f'%b, ha='center', va='bottom', rotation=90)

ax[0].set_title('Bit-Level Sparsity', fontsize=font_big, fontweight='bold')
ax[0].set_xlabel('Model', fontsize=font_mid, fontweight='bold')
ax[0].set_ylabel('Percentage (%)', fontsize=font_mid, fontweight='bold')
leg = ax[0].legend(fontsize=font_legend)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(2)

ax[0].xaxis.set_tick_params(labelsize=font_small_x)
ax[0].yaxis.set_tick_params(labelsize=font_small_y)

ax[0].set_xticks(x + bar_width*(len(data_bit)-2)/2)
ax[0].set_xticklabels(tick_labels, rotation=20)

ax[0].spines['bottom'].set_linewidth(axis_width)
ax[0].spines['left'].set_linewidth(axis_width)
ax[0].spines['top'].set_linewidth(axis_width)
ax[0].spines['right'].set_linewidth(axis_width)



for i in range(len(data_booth)):
    ax[1].bar(x + i*bar_width, data_booth[i], width=bar_width, label='#1 = '+str(i))
    if i < 7:
        for a, b in zip(x, data_booth[i]):
            ax[1].text(a + i*bar_width+0.01, b+0.3, '%.1f'%b, ha='center', va='bottom', rotation=90)

ax[1].set_title('Bit-Level Sparsity \n with 4-bit Booth Encoding', fontsize=font_big, fontweight='bold')
ax[1].set_xlabel('Model', fontsize=font_mid, fontweight='bold')
ax[1].set_ylabel('Percentage (%)', fontsize=font_mid, fontweight='bold')
leg = ax[1].legend(fontsize=font_legend, loc=2, bbox_to_anchor=(0.26, 1))
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(2)

ax[1].xaxis.set_tick_params(labelsize=font_small_x)
ax[1].yaxis.set_tick_params(labelsize=font_small_y)

ax[1].set_xticks(x + bar_width*(len(data_booth)-1)/2)
ax[1].set_xticklabels(tick_labels, rotation=20)

ax[1].spines['bottom'].set_linewidth(axis_width)
ax[1].spines['left'].set_linewidth(axis_width)
ax[1].spines['top'].set_linewidth(axis_width)
ax[1].spines['right'].set_linewidth(axis_width)

plt.savefig('activation_sparsity.pdf', bbox_inches='tight')



 