from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class DatasetBERT(Dataset):
    def __init__(self, modality1_data, modality2_data, modality3_data, label_data):

        self.modality1 = modality1_data.astype(np.float32)
        self.modality2 = modality2_data.astype(np.float32)
        self.modality3 = modality3_data.astype(np.float32)
        self.label = label_data.astype(np.int64)
        self.n_samples = modality1_data.shape[0]
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):

        sample = self.modality1[index], self.modality2[index], self.modality3[index], self.label[index]

        return sample

