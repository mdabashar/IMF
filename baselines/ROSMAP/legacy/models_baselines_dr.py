import torch.nn as nn
import torch
import torch.nn.functional as F

class Unimodal(nn.Module):
    def __init__(self, length = 200, input_length=200, functional_p = 0.3):
        super().__init__()
        self.drop = nn.Dropout(p=functional_p)
        self.early = nn.Linear(length, 2)
        self.trans = nn.Linear(input_length, length)
    def forward(self, md):
        md = F.relu(self.trans(md))
        clf = self.early(md)
        return clf


class Early_fusion(nn.Module):
    def __init__(self, length = 200, input_length=200, functional_p = 0.3):
        super().__init__()
        self.drop = nn.Dropout(p=functional_p)
        self.trans_m3 = nn.Linear(200, length)
        self.trans_m2 = nn.Linear(200, length)
        self.trans_m1 = nn.Linear(200, length)
        self.early = nn.Linear(length, 2)

    def forward(self, m1d, m2d, m3d):
        m1d = F.relu(self.trans_m1(m1d))
        m2d = F.relu(self.trans_m2(m2d))
        m3d = F.relu(self.trans_m3(m3d)) 
        cat_feature = torch.add(m1d.float(), m2d.float())
        cat_feature = torch.add(cat_feature.float(), m3d.float())
        clf = self.early(cat_feature)
        return clf


class Late_fusion(nn.Module):
    def __init__(self, length = 200, input_length=200, functional_p = 0.3):
        super().__init__()
        self.trans_m3 = nn.Linear(200, length)
        self.trans_m2 = nn.Linear(200, length)
        self.trans_m1 = nn.Linear(200, length)
        self.late1 = nn.Linear(length, 2)
        self.late2 = nn.Linear(length, 2)
        self.late3 = nn.Linear(length, 2)
        self.dropout = nn.Dropout(p=functional_p) 

    def forward(self, m1d, m2d, m3d):
        
        m1d = F.relu(self.trans_m1(m1d))
        m2d = F.relu(self.trans_m2(m2d))
        m3d = F.relu(self.trans_m3(m3d))
        m1d = self.dropout(self.late1(m1d))
        m2d = self.dropout(self.late2(m2d))
        m3d = self.dropout(self.late3(m3d))
        clf = (m1d + m2d + m3d)/3
        return clf

class GMU(nn.Module):
    def __init__(self, length = 200, input_length=200, functional_p = 0.3):
        super().__init__()
        self.gate1 = nn.Linear(3*length, length)
        self.gate2 = nn.Linear(3*length, length)
        self.gate3 = nn.Linear(3*length, length)
        self.drop = nn.Dropout(p=functional_p)
        self.early = nn.Linear(length, 2)
        self.sigmoid = nn.Sigmoid()
        self.trans_m3 = nn.Linear(200, length)
        self.trans_m2 = nn.Linear(200, length)
        self.trans_m1 = nn.Linear(200, length)

    def forward(self, m1d, m2d, m3d):
         
        m1d = F.relu(self.trans_m1(m1d))
        m2d = F.relu(self.trans_m2(m2d))
        m3d = F.relu(self.trans_m3(m3d))

        gate1 = self.sigmoid(self.drop(self.gate1(torch.cat([m1d, m2d, m3d], dim=1))))
        m1d = m1d*gate1 
        gate2 = self.sigmoid(self.drop(self.gate2(torch.cat([m1d, m2d, m3d], dim=1))))
        m2d = m2d*gate2 
        gate3 = self.sigmoid(self.drop(self.gate3(torch.cat([m1d, m2d, m3d], dim=1))))
        m3d = m3d*gate3 
        cat_feature = torch.add(m1d.float(), m2d.float())
        cat_feature = torch.add(cat_feature.float(), m3d.float())
        clf = self.early(cat_feature)
        return clf

class Cross(nn.Module):
    def __init__(self, length = 200, input_length=200, functional_p = 0.3):
        super().__init__()
        self.gate1 = nn.Linear(2*length, length)
        self.gate2 = nn.Linear(2*length, length)
        self.gate3 = nn.Linear(2*length, length)
        self.drop = nn.Dropout(p=functional_p)
        self.early = nn.Linear(length, 2)
        self.sigmoid = nn.Sigmoid() 
        self.trans_m3 = nn.Linear(200, length)
        self.trans_m2 = nn.Linear(200, length)
        self.trans_m1 = nn.Linear(200, length)

    def forward(self, m1d, m2d, m3d):

        m1d = F.relu(self.trans_m1(m1d))
        m2d = F.relu(self.trans_m2(m2d))
        m3d = F.relu(self.trans_m3(m3d))

        gate1 = self.sigmoid(self.drop(self.gate1(torch.cat([m2d, m3d], dim=1))))
        m1d = m1d*gate1
        gate2 = self.sigmoid(self.drop(self.gate2(torch.cat([m1d, m3d], dim=1))))
        m2d = m2d*gate2
        gate3 = self.sigmoid(self.drop(self.gate3(torch.cat([m1d, m2d], dim=1))))
        m3d = m3d*gate3
        cat_feature = torch.add(m1d.float(), m2d.float())
        cat_feature = torch.add(cat_feature.float(), m3d.float())
        clf = self.early(cat_feature)
        return clf
