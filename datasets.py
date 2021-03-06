import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train', size=(1, 1, 256, 256)):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.size = size;
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B1 = sorted(glob.glob(os.path.join(root, '%s/B1' % mode) + '/*.*'))
        self.files_B2 = sorted(glob.glob(os.path.join(root, '%s/B2' % mode) + '/*.*'))

    def __getitem__(self, index):
        file_A = self.files_A[index % len(self.files_A)]
        item_A = self.transform(Image.open(file_A))
        
        if self.unaligned:

            file_B1 = self.files_B1[random.randint(0, len(self.files_B1) -1)]
            item_B1 = self.transform(Image.open(file_B1))
            file_B2 = self.files_B2[random.randint(0, len(self.files_B2) -1)]
            item_B2 = self.transform(Image.open(file_B2))
        else:
            file_B1 = self.files_B1[index % len(self.files_B1)]
            item_B1 = self.transform(Image.open(file_B1))
            file_B2 = self.files_B2[index % len(self.files_B2)]
            item_B2 = self.transform(Image.open(file_B2))

        return {'A': item_A, 'B1': item_B1, 'B2': item_B2, 'FA':file_A, 'FB1':file_B1, 'FB2':file_B2}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B1), len(self.files_B2))
