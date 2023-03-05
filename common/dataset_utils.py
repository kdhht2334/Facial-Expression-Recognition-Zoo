import os
import pandas as pd
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset


class FaceDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, inFolder=None, landmarks=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.training_sheet = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        if inFolder is None:
            self.inFolder = np.full((len(self.training_sheet),), True)
        
        self.loc_list = np.where(inFolder)[0]
        self.infold = self.inFolder
        self.balance_factor = 1.0
        
    def __len__(self):
        return  np.sum(self.infold*1)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.training_sheet.iloc[idx, 0])
        valence = self.training_sheet.iloc[idx,1] * self.balance_factor
        arousal = self.training_sheet.iloc[idx,2] * self.balance_factor
        
        sample = Image.open(img_name)
        
        if self.transform:
            sample = self.transform(sample)
        return {'image': sample, 'va': [valence, arousal], 'path': self.training_sheet.iloc[idx, 0]}


class AddGaussianNoise(object):

    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
        
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
