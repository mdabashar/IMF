import torch.nn as nn
import torch
import torch.nn.functional as F
from unimodals_gp import *
class EarlyFusion(nn.Module):
    def __init__(self, length = 200, input_length=200, functional_p = 0.3):
        super().__init__()
        self.trans_m2 = nn.Linear(length, length)
        self.trans_m1 = nn.Linear(length, length)
        self.late1 = nn.Linear(length, 101)
        self.late2 = nn.Linear(length, 101)
        self.late3 = nn.Linear(length, 101)
        self.dropout = nn.Dropout(p=functional_p) 

        self.encoders0 = nn.Linear(768, length)
        self.encoders1 = nn.Linear(768, length)

    def forward(self, m1d, m2d):
        m1d = self.encoders0(m1d)
        m2d = self.encoders1(m2d) 
        m1d = self.dropout(F.relu(self.trans_m1(m1d)))
        m2d = self.dropout(F.relu(self.trans_m2(m2d)))
        m1d = self.late1(m1d)
        m2d = self.late2(m2d)
        clf = (m1d + m2d )/2
        return [clf,m1d,m2d]

