import torch.nn as nn
import torch
import torch.nn.functional as F

class Unimodal(nn.Module):
    def __init__(self,  length = 200, input_length=768, functional_p = 0.3):
        super().__init__()
        self.modality = nn.Linear(input_length, length)
        self.drop = nn.Dropout(p=functional_p)
        self.early = nn.Linear(input_length, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, md):
        #md = torch.relu(self.modality(md))
        clf = self.sigmoid(self.early(md))
        return clf

class Early_fusion(nn.Module):
    def __init__(self, length = 200, input_length=200, functional_p = 0.3):
        super().__init__()
        self.drop = nn.Dropout(p=functional_p)
        self.early = nn.Linear(768+2048+283, 1)
        self.sigmoid = nn.Sigmoid()
        self.norm1 = nn.BatchNorm1d(768)
        self.norm2 = nn.BatchNorm1d(2048)
        self.norm3 = nn.BatchNorm1d(283)
    def forward(self, m1d, m2d, m3d):
        m1d = self.norm1(m1d)
        m2d = self.norm2(m2d)
        m3d = self.norm3(m3d)
        cat_feature = torch.cat([m1d.float(), m2d.float()],dim=1)
        cat_feature = torch.cat([cat_feature.float(), m3d.float()], dim=1)
        clf = self.sigmoid(self.early(cat_feature))
        return clf




class Add_fusion(nn.Module):
    def __init__(self, length = 200, input_length=200, functional_p = 0.3):
        super().__init__()
        self.drop = nn.Dropout(p=functional_p)
        self.early = nn.Linear(length, 1)
        self.sigmoid = nn.Sigmoid()
        self.modality1 = nn.Linear(768, length)
        self.modality2 = nn.Linear(2048, length)
        self.modality3 = nn.Linear(283, length)
        self.norm = nn.BatchNorm1d(length)
    def forward(self, m1d, m2d, m3d):

        m1d = torch.relu(self.modality1(m1d))
        m2d = torch.relu(self.modality2(m2d))
        m3d = torch.relu(self.modality3(m3d))
        m1d = self.norm(m1d)
        m2d = self.norm(m2d)
        m3d = self.norm(m3d)
        cat_feature = torch.add(m1d.float(), m2d.float())
        cat_feature = torch.add(cat_feature.float(), m3d.float())
        clf = self.sigmoid(self.early(cat_feature))
        return clf


class Late_fusion(nn.Module):
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
        return clf

class GMU(nn.Module):
    def __init__(self, length = 200, input_length=200, functional_p = 0.3):
        super().__init__()

        self.modality1 = nn.Linear(768, length)
        self.modality2 = nn.Linear(2048, length)
        self.modality3 = nn.Linear(283, length)

        self.gate1 = nn.Linear(3*length, length)
        self.gate2 = nn.Linear(3*length, length)
        self.gate3 = nn.Linear(3*length, length)
        self.drop = nn.Dropout(p=functional_p)
        self.early = nn.Linear(length, 1)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm1d(length)
    def forward(self, m1d, m2d, m3d):

        m1d = torch.relu(self.modality1(m1d))
        m2d = torch.relu(self.modality2(m2d))
        m3d = torch.relu(self.modality3(m3d))

        m1d = self.norm(m1d)
        m2d = self.norm(m2d)
        m3d = self.norm(m3d)

        gate1 = self.sigmoid(self.drop(self.gate1(torch.cat([m1d, m2d, m3d], dim=1))))
        m1d = m1d*gate1 
        gate2 = self.sigmoid(self.drop(self.gate2(torch.cat([m1d, m2d, m3d], dim=1))))
        m2d = m2d*gate2 
        gate3 = self.sigmoid(self.drop(self.gate3(torch.cat([m1d, m2d, m3d], dim=1))))
        m3d = m3d*gate3 
        cat_feature = torch.add(m1d.float(), m2d.float())
        cat_feature = torch.add(cat_feature.float(), m3d.float())
        clf = self.sigmoid(self.early(cat_feature))
        return clf

class Cross(nn.Module):
    def __init__(self, length = 200, input_length=200, functional_p = 0.3):
        super().__init__()
        self.modality1 = nn.Linear(768, length)
        self.modality2 = nn.Linear(2048, length)
        self.modality3 = nn.Linear(283, length)

        self.gate1 = nn.Linear(2*length, length)
        self.gate2 = nn.Linear(2*length, length)
        self.gate3 = nn.Linear(2*length, length)
        self.drop = nn.Dropout(p=functional_p)
        self.early = nn.Linear(length, 1)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm1d(length)
    def forward(self, m1d, m2d, m3d):

        m1d = torch.relu(self.modality1(m1d))
        m2d = torch.relu(self.modality2(m2d))
        m3d = torch.relu(self.modality3(m3d))

        m1d = self.norm(m1d)
        m2d = self.norm(m2d)
        m3d = self.norm(m3d)

        gate1 = self.sigmoid(self.drop(self.gate1(torch.cat([m2d, m3d], dim=1))))
        m1d = m1d*gate1
        gate2 = self.sigmoid(self.drop(self.gate2(torch.cat([m1d, m3d], dim=1))))
        m2d = m2d*gate2
        gate3 = self.sigmoid(self.drop(self.gate3(torch.cat([m1d, m2d], dim=1))))
        m3d = m3d*gate3
        cat_feature = torch.add(m1d.float(), m2d.float())
        cat_feature = torch.add(cat_feature.float(), m3d.float())
        clf = self.sigmoid(self.early(cat_feature))
        return clf

class Tensor(nn.Module):
    def __init__(self, length = 200, input_length=200, functional_p = 0.3):
        super().__init__()
        self.drop = nn.Dropout(p=functional_p)
        self.sigmoid = nn.Sigmoid()
        self.length = length
        self.modality1 = nn.Linear(768, length)
        self.modality2 = nn.Linear(2048, length)
        self.modality3 = nn.Linear(283, length)
        self.norm = nn.BatchNorm1d(length)
        self.norm_2 = nn.BatchNorm2d(length, affine=False)

        self.early_1 = nn.Linear((length+1)*(length+1)*(length+1), length+1)
        self.early_2 = nn.Linear(length+1, length+1)
        self.early = nn.Linear(length+1, 1)
    def forward(self, m1d, m2d, m3d):

        batch_size = m1d.size()[0]
        m1d = torch.relu(self.modality1(m1d))
        m2d = torch.relu(self.modality2(m2d))
        m3d = torch.relu(self.modality3(m3d))

        #m1d = self.norm(m1d)
        #m2d = self.norm(m2d)
        #m3d = self.norm(m3d)

        m1d = torch.cat([m1d, torch.ones(m1d.size()[0],1)], dim=1)
        m2d = torch.cat([m2d, torch.ones(m2d.size()[0],1)], dim=1)
        m3d = torch.cat([m3d, torch.ones(m3d.size()[0],1)], dim=1)

        cat_feature = torch.bmm(m1d.unsqueeze(2),m2d.unsqueeze(1))
        #cat_feature = self.norm(cat_feature)
        cat_feature = torch.bmm(cat_feature.view(-1,(self.length+1)*(self.length+1),1),m3d.unsqueeze(1)).view(batch_size, -1)
        #cat_feature = self.norm(cat_feature)
        cat_feature = torch.relu(self.early_1(cat_feature))
        #cat_feature = self.norm(cat_feature)
        cat_feature = torch.relu(self.early_2(cat_feature))
        #cat_feature = self.norm(cat_feature)
        clf = self.sigmoid(self.early(cat_feature))
        return clf

class TensorGate(nn.Module):
    def __init__(self, length = 200, gate = "im", input_length=200, functional_p = 0.3):
        super().__init__()
        self.drop = nn.Dropout(p=functional_p)
        self.sigmoid = nn.Sigmoid()
        self.length = length
        self.gate = gate
        self.modality1 = nn.Linear(768, length)
        self.modality2 = nn.Linear(2048, length)
        self.modality3 = nn.Linear(283, length)
        self.norm = nn.BatchNorm1d(length)
        self.norm_2 = nn.BatchNorm2d(length, affine=False)

        self.early_1 = nn.Linear((length+1)*(length+1)*(length+1), length+1)
        self.early_2 = nn.Linear(length+1, length+1)
        self.early = nn.Linear(length+1, 1)

    def forward(self, m1d, m2d, m3d):

        batch_size = m1d.size()[0]
        m1d = torch.relu(self.modality1(m1d))
        m2d = torch.relu(self.modality2(m2d))
        m3d = torch.relu(self.modality3(m3d))

        if self.gate == "m1":
           m1d = torch.sigmoid(m1d)
           m2d = torch.relu(m2d)
           m3d = torch.relu(m3d)
        elif self.gate == "m2":
           m1d = torch.relu(m1d)
           m2d = torch.sigmoid(m2d)
           m3d = torch.relu(m3d)
        elif self.gate == "m3":
           m1d = torch.relu(m1d)
           m2d = torch.relu(m2d)
           m3d = torch.sigmoid(m3d)

        m1d = torch.cat([m1d, torch.ones(m1d.size()[0],1)], dim=1)
        m2d = torch.cat([m2d, torch.ones(m2d.size()[0],1)], dim=1)
        m3d = torch.cat([m3d, torch.ones(m3d.size()[0],1)], dim=1)

        cat_feature = torch.bmm(m1d.unsqueeze(2),m2d.unsqueeze(1))
        cat_feature = torch.bmm(cat_feature.view(-1,(self.length+1)*(self.length+1),1),m3d.unsqueeze(1)).view(batch_size, -1)
  
        cat_feature = torch.relu(self.early_1(cat_feature))
        cat_feature = torch.relu(self.early_2(cat_feature))
        clf = self.sigmoid(self.early(cat_feature))
        return clf

