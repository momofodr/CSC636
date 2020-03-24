import glob
import random
import os
import numpy as np
import sys
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import re

class ImageDataset(Dataset):
    def __init__(self, root, labels_, transforms_=None, mode='train'):
        
        self.files = []
        self.labels = labels_
        for img in root:
            self.files.append(transforms_(Image.fromarray(img)))

    def __getitem__(self, index):
        img = self.files([index])
        label = int(self.labels[index])

        return img, label

    def __len__(self):
        return len(self.files)




if __name__ == '__main__':
    dataset_path = sys.argv[1]
    crop_img_size = 128

    # original img shape (288, 384, 3)
    transforms_ = [ transforms.RandomCrop(crop_img_size),
                    transforms.ToTensor()]

    train_loader_model = torch.utils.data.DataLoader(ImageDataset(dataset_path, transforms_=transforms_, mode='train'), 
                        batch_size=8, shuffle=True, num_workers=4)

    for i, (img, label) in enumerate(train_loader_model):
        print(label)
        save_image(img, './%d.jpg' % i)
        exit()
