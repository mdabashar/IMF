from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import numpy as np


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

def getDataloder(fold = 0):
    modality3_data = np.load("../../../datasets/mustard/mustard_audio.npy").astype(np.float32)
    modality2_data = np.load("../../../datasets/mustard/mustard_video.npy").astype(np.float32)
    modality1_data = np.load("../../../datasets/mustard/mustard_text.npy").astype(np.float32)
    y_data = np.argmax(np.load("../../../datasets/mustard/mustard_label.npy"), axis=1)
    x_data = np.arange(modality1_data.shape[0])
    # 999, 233
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=999)

    for i, (tr_index, te_index) in enumerate(skf.split(x_data, y_data)):
        if i == fold:
            train_index = tr_index
            test_index = te_index
            break

    train_dataset = (modality1_data[train_index], modality2_data[train_index], modality3_data[train_index], y_data[train_index])
    test_dataset = DatasetBERT(modality1_data[test_index], modality2_data[test_index], modality3_data[test_index], y_data[test_index])
    return train_dataset, test_dataset

