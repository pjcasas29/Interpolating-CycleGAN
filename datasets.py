import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B1 = sorted(glob.glob(os.path.join(root, '%s/B1' % mode) + '/*.*'))
        self.files_B2 = sorted(glob.glob(os.path.join(root, '%s/B2' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B1 = self.transform(Image.open(self.files_B1[random.randint(0, len(self.files_B1) - 1)]))
            item_B2 = self.transform(Image.open(self.files_B2[random.randint(0, len(self.files_B2) - 1)]))
        else:
            item_B1 = self.transform(Image.open(self.files_B1[index % len(self.files_B1)]))
            item_B2 = self.transform(Image.open(self.files_B2[index % len(self.files_B2)]))

        return {'A': item_A, 'B1': item_B1, 'B2': item_B2}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B1), len(self.files_B2))
