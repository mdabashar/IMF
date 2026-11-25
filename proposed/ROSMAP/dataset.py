from torch.utils.data import Dataset
import numpy as np
import pandas as pd
class DatasetMul(Dataset):
    def __init__(self, im_dir, txt_dir, y_dir, ifyelp=False):
        self.im = np.load(im_dir, allow_pickle=True)
        self.txt = np.load(txt_dir, allow_pickle=True)
        self.y = np.int64(np.load(y_dir, allow_pickle=True))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):

        sample = self.im[index][0], self.txt[index],self.y[index]

        return sample

    def getTextdim(self):
        return self.txt.shape[1]

    def getImagedim(self):
        return self.im.shape[1]

    def dist(self):
        labels = np.unique(self.y)
        class_dist = {x: (self.y==x).sum() for x in [0,1,2]}
        return class_dist

class DatasetBERT(Dataset):
    def __init__(self, modality1_data, modality2_data, modality3_data, label_data):
        self.modality1 = modality1_data
        self.modality2 = modality2_data
        self.modality3 = modality3_data
        self.label = label_data
        self.n_samples = modality1_data.shape[0]
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):

        sample = self.modality1[index], self.modality2[index], self.modality3[index],self.label[index]

        return sample
