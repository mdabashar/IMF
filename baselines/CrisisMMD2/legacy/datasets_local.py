from torch.utils.data import Dataset
import numpy as np

class Crisismmd(Dataset):
    def __init__(self, x_dir, y_dir, is_txt):
        if is_txt:
            self.x = np.load(x_dir, allow_pickle=True).astype("int64")
        else:
            self.x = np.load(x_dir, allow_pickle=True).astype("float32")
        self.y = np.int64(np.load(y_dir, allow_pickle=True))
        self.n_samples = self.x.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        return sample

class CrisismmdViT(Dataset):
    def __init__(self, x_dir, y_dir, is_txt):
        if is_txt:
            self.x = np.load(x_dir, allow_pickle=True).astype("int64")
        else:
            self.x = x_dir
        self.y = np.int64(np.load(y_dir, allow_pickle=True))
        self.n_samples = self.x.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        return sample

class DatasetMul(Dataset):
    def __init__(self, im_dir, txt_dir, y_dir):
        self.im = np.load(im_dir, allow_pickle=True)
        self.txt = np.load(txt_dir, allow_pickle=True).astype("float32")
        self.y = np.int64(np.load(y_dir, allow_pickle=True))
        self.n_samples = self.im.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):

        sample = self.im[index][0], self.txt[index],self.y[index]

        return sample

    def getTextdim(self):
        return self.txt.shape[1]

    def dist(self):
        labels = np.unique(self.y)
        class_dist = {x: (self.y==x).sum() for x in [0,1,2]}
        return class_dist


class DatasetBERT(Dataset):
    def __init__(self, im_data, txt_data, y_data):
        self.im = im_data
        self.txt = txt_data.astype("float32")
        self.y = y_data
        self.n_samples = self.im.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):

        sample = self.im[index], self.txt[index],self.y[index]

        return sample

    def dist(self):
        labels = np.unique(self.y)
        class_dist = {x: (self.y==x).sum() for x in [0,1,2]}
        return class_dist
