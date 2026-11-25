import torch.nn as nn
import torch
import torch.nn.functional as F
from unimodals_gp import *
class Early_fusion(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.early = nn.Linear(2*768, 101)
        self.lr = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)
    def forward(self,im, txt, sigma_txt=0, sigma_im=0):
        
        cat_feature = torch.cat([im.float(),txt.float()], dim = 1)
        clf = self.early(cat_feature)
        return clf

class Early_fusion2(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()

        self.im_dim = nn.Linear(768, length, bias=False)
        self.text_dim = nn.Linear(768, length, bias=False)
        self.drop = nn.Dropout(p=functional_p)
        self.early = nn.Linear(length, 101)
    def forward(self,im, txt, sigma_txt=0, sigma_im=0):
        im_embed = torch.relu(self.drop(self.im_dim(im))) ###[batch_size, 512, 1]
        txt_embed = torch.relu(self.drop(self.text_dim(txt)))
        cat_feature = torch.add(im_embed.float(),txt_embed.float())
        clf = self.early(cat_feature)
        return clf


class Late_fusion(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()
        self.im_dim = nn.Linear(768, 101, bias=False)
        self.text_dim = nn.Linear(768, 101, bias=False)

        #self.im_dim_4 = nn.Linear(length, 101)
        #self.text_dim_4 = nn.Linear(length, 101)
        self.drop = nn.Dropout(p=functional_p)

    def forward(self,im, txt, sigma_txt=0, sigma_im=0):

        im_embed = torch.relu(self.drop(self.im_dim(im))) ###[batch_size, 512, 1]
        #im_embed = torch.relu(self.drop(self.im_dim_4(im_embed))) ###[batch_size, 512, 1]

        txt_embed = torch.relu(self.drop(self.text_dim(txt)))
        #txt_embed = torch.relu(self.drop(self.text_dim_4(txt_embed)))

        clf = torch.add(im_embed, txt_embed)/2
        return clf

class Cross(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()

        self.early = nn.Linear(2*length, 101)
        self.encoders0 = nn.Linear(768, length)
        self.encoders1 = nn.Linear(768, length)
        self.im_sigma = nn.Linear(length, length, bias=False)
        self.text_sigma = nn.Linear(length, length, bias=False)
        self.drop = nn.Dropout(p=functional_p)

        self.early = nn.Linear(length, 101)
    def forward(self,im, txt, sigma_txt=0, sigma_im=0):

        im_embed = self.encoders0(im)
        txt_embed = self.encoders1(txt)

        sigma_txt = torch.sigmoid(self.drop(self.text_sigma(im_embed))) ###[batch_size, 512, 1]
        sigma_im = torch.sigmoid(self.drop(self.im_sigma(txt_embed))) ###[batch_size, 512, 1]

        im_embed = im_embed*sigma_im
        txt_embed = txt_embed*sigma_txt

        cat_feature = torch.add(im_embed.float(),txt_embed.float())
        clf = self.early(cat_feature)
        return clf

class GMU(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()

        self.im_dim = nn.Linear(768, length, bias=False)
        self.text_dim = nn.Linear(768, length, bias=False)

        self.sigma = nn.Linear(2*length, length, bias=False)
        self.drop = nn.Dropout(p=functional_p)
        self.early = nn.Linear(length, 101)
    def forward(self,im, txt, sigma_txt=0, sigma_im=0):

        im_embed = torch.tanh(self.drop(self.im_dim(im))) ###[batch_size, 512, 1]
        txt_embed = torch.tanh(self.drop(self.text_dim(txt)))

        weight = torch.sigmoid(self.sigma(torch.cat([txt_embed, im_embed], dim=1)))

        cat_feature = torch.add(im_embed.float()* weight,txt_embed.float()*(1-weight))
        clf = self.early(cat_feature)
        return clf
class Tensor(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()
        self.im_dim = nn.Linear(768, length, bias=False)
        self.text_dim = nn.Linear(768, length, bias=False)

        self.drop = nn.Dropout(p=functional_p)

        self.early_1 = nn.Linear((length+1)*(length+1), length+1)
        self.early = nn.Linear(length+1, 101)
    def forward(self,im, txt):

        im_embed = torch.relu(self.drop(self.im_dim(im)))

        txt_embed = torch.relu(self.drop(self.text_dim(txt)))

        im_embed = torch.cat([im_embed, torch.ones(im_embed.size()[0],1)], dim=1)
        txt_embed = torch.cat([txt_embed,  torch.ones(im_embed.size()[0],1)], dim=1)

        cat_feature = torch.bmm(im_embed.unsqueeze(2),txt_embed.unsqueeze(1)).view(im_embed.size()[0], -1)
        #print(cat_feature.size())
        cat_feature = torch.relu(self.early_1(cat_feature))
        #print(cat_feature.size())
        clf = self.early(cat_feature)
        return clf

class TensorGate(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768, gate="im",
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()
        self.im_dim = nn.Linear(768, length, bias=False)
        self.text_dim = nn.Linear(768, length, bias=False)
        self.gate = gate
        self.drop = nn.Dropout(p=functional_p)

        self.early_1 = nn.Linear((length+1)*(length+1), length+1)
        self.early = nn.Linear(length+1, 101)
    def forward(self,im, txt):

        im_embed = torch.relu(self.drop(self.im_dim(im)))
        txt_embed = torch.relu(self.drop(self.text_dim(txt)))

        if self.gate == "im":
           im_embed = torch.sigmoid(im_embed)
           txt_embed = torch.relu(txt_embed)
        else:
           im_embed = torch.relu(im_embed)
           txt_embed = torch.sigmoid(txt_embed)

        im_embed = torch.cat([im_embed, torch.ones(im_embed.size()[0],1)], dim=1)
        txt_embed = torch.cat([txt_embed,  torch.ones(im_embed.size()[0],1)], dim=1)

        cat_feature = torch.bmm(im_embed.unsqueeze(2),txt_embed.unsqueeze(1)).view(im_embed.size()[0], -1)
        #print(cat_feature.size())
        cat_feature = torch.relu(self.early_1(cat_feature))
        #print(cat_feature.size())
        clf = self.early(cat_feature)
        return clf


class Text(nn.Module):
    def __init__(self, length = 20, functional_p = 0.3):
        super().__init__()
        self.encoders = nn.Linear(768, length)

        self.early = nn.Linear(length, 101)
        self.drop = nn.Dropout(p=functional_p)
        self.lr = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)
    def forward(self,txt, sigma_txt=0, sigma_im=0):
        
        txt_embed = self.lr(self.encoders(txt))
        txt_embed = self.early(txt_embed)

        return txt_embed


class Image(nn.Module):
    def __init__(self, length = 20, functional_p = 0.3):
        super().__init__()
        self.encoders = nn.Linear(768,length)
        self.early = nn.Linear(768, 101)
        self.drop = nn.Dropout(p=functional_p)
        #self.lr = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.lr = torch.nn.ReLU()
    def forward(self,im, sigma_txt=0, sigma_im=0):

        #im_embed = self.lr(self.encoders(im)) 
        im_embed = self.early(im)

        return im_embed
