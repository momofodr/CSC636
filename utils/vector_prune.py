import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_masks(model, pruning_ratio):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''    
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) == 4:
            all_weights += list(torch.sum(p.data.abs(), dim=3).cpu().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), 100-pruning_ratio)

    # generate mask
    masks = []
    for p in model.parameters():
        if len(p.data.size()) == 4:
            mask = (torch.sum(p.data.abs(), dim=3) > threshold).float()
            masks.append(mask)

    return masks


def set_masks(model, masks):
    i = 0
    for p in model.parameters():
        if len(p.data.size()) == 4:
            p.data = p.data * torch.unsqueeze(masks[i], -1)
            i += 1


def count_sparsity(model):
    i = 1
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            sparsity = torch.sum(module.weight == 0).float() / module.weight.numel()
            print('layer%d:' % i, sparsity)
            i += 1