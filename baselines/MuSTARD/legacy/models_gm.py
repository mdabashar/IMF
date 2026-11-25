import torch.nn as nn
import torch
import torch.nn.functional as F

class EarlyFusionModel(nn.Module):
    def __init__(self, length = 200, input_length=200, functional_p = 0.3):
        super().__init__()

        self.modality1 = nn.Linear(768, length)
        self.modality2 = nn.Linear(2048, length)
        self.modality3 = nn.Linear(283, length)

        self.late1 = nn.Linear(length, 1)
        self.late2 = nn.Linear(length, 1)
        self.late3 = nn.Linear(length, 1)
        self.dropout = nn.Dropout(p=functional_p) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, m1d, m2d, m3d):

        m1d = torch.relu(self.modality1(m1d))
        m2d = torch.relu(self.modality2(m2d))
        m3d = torch.relu(self.modality3(m3d))
        m1d = self.sigmoid(self.dropout(self.late1(m1d)))
        m2d = self.sigmoid(self.dropout(self.late2(m2d)))
        m3d = self.sigmoid(self.dropout(self.late3(m3d)))
        clf = (m1d + m2d + m3d)/3
        return [clf, m1d, m2d, m3d]

