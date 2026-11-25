import torch.nn as nn
import torch
import torch.nn.functional as F
from unimodals_gp import *

def uncertaintyNormalization(uncertainty_im, uncertainty_txt):
    n_im = (1- (uncertainty_im + (1e-15))/(uncertainty_im + uncertainty_txt + 2*(1e-15)))
    n_txt = (1- (uncertainty_txt + (1e-15))/(uncertainty_im + uncertainty_txt + 2*(1e-15)))
    # print("BEFORE: ", uncertainty_im)
    # print("AFTER: ", n_im)
    return n_im, n_txt


class Meta_net(nn.Module):
    def __init__(self, length = 100, bnn_p = 0.5, other_dim = 40):
        super(Meta_net, self).__init__()
        self.dropout = nn.Dropout(p=bnn_p)
        self.encoder1 = nn.Linear(768, length)
        self.encoder2 = nn.Linear(768, length)
    def forward(self, x):

        mu = self.dropout(self.encoder1(x))

        sigma = self.dropout(self.encoder2(x))
        
        return [mu, sigma]

class Meta_net_txt(nn.Module):
    def __init__(self, input_dim, input_length, dim = 10,
                 isImage=False, length = 100, bnn_p = 0.5, other_dim = 40):
        super(Meta_net_txt, self).__init__()
        self.dropout = nn.Dropout(p=bnn_p)
        self.fc = nn.Linear(input_dim, length)

        self.fc_2 = nn.Linear(input_dim, length)



    def forward(self, x):

        x= torch.mean(x,1)

        mu = torch.relu(self.fc(x))  ### [batch_size, 512, 30]

        sigma = torch.relu(self.fc_2(x))  ### [batch_size, 512, 30]

        return [mu, sigma]


class Meta_net_cat(nn.Module):
    def __init__(self, length = 100, bnn_p = 0.5, txt_dim=40, im_dim=512):
        super(Meta_net_cat, self).__init__()
        self.dropout = nn.Dropout(p=bnn_p)
        self.fc = nn.Linear(length, length)
        # self.norm = nn.BatchNorm1d(int(length/2))

        self.fc_2 = nn.Linear(length, length)
        # self.norm_2 = nn.BatchNorm1d(int(length/2))

    def forward(self, x):
        # mu = self.norm(self.dropout(torch.relu(self.fc(x))))
        # sigma = self.norm_2(self.dropout(torch.relu(self.fc_2(x))))
        mu = torch.relu(self.dropout(self.fc(x)))
        sigma = torch.relu(self.dropout(self.fc_2(x)))
        return [mu, sigma]


class Early_fusion_new_0(nn.Module):
    def __init__(self, length = 20, dim=10, functional_p = 0.3):
        super().__init__()

        self.m1_norm = nn.BatchNorm1d(length)
        self.m2_norm = nn.BatchNorm1d(length)
        self.cat_norm = nn.BatchNorm1d(length)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoders0 = nn.Linear(768, length)
        self.encoders1 = nn.Linear(768, length)
        self.leakyRelu = torch.nn.LeakyReLU(negative_slope=0.1)
        #self.proj_m1 = nn.Linear(length, length)
        #self.proj_m2 = nn.Linear(length, length)
        # self.im_weight = nn.Linear(length, length, bias=False)
        # self.text_weight = nn.Linear(length, length, bias=False)
        self.drop = nn.Dropout(p=functional_p)
        self.pre_early_2 = nn.Linear(length, length)
        #self.pre_early_3 = nn.Linear(int(length/2), int(length/4))
        self.early = nn.Linear(length, 101)
        self.softplus = torch.nn.Softplus(beta=1)

    def forward(self,m1d,m2d, sigma_m1, sigma_m2, model_cat):

        mean_m1, var_m1 = sigma_m1
        #mean_im = torch.clamp(mean_im, min=-5, max=5)
        var_m1 = torch.clamp(var_m1, min=0, max=1)
        sample_m1 = torch.normal(mean= torch.zeros(mean_m1.size()), std= torch.ones(mean_m1.size())).to(self.device)
        
        mean_m2, var_m2 = sigma_m2
        #mean_txt = torch.clamp(mean_txt, min=-5, max=5)
        var_m2 = torch.clamp(var_m2, min=0, max=1)
        sample_m2 = torch.normal(mean= torch.zeros(mean_m2.size()), std= torch.ones(mean_m2.size())).to(self.device)
        
        m1d = self.encoders0(m1d)
        m2d = self.encoders1(m2d)

        m1_embed = self.m1_norm(m1d)
        #m1_embed = m1d
        m1_embed = torch.add(m1_embed, self.softplus(mean_m1 + var_m1 *sample_m1))
        m2_embed = self.m2_norm(m2d)
        #m2_embed = m2d
        m2_embed = torch.add(m2_embed, self.softplus(mean_m2 + var_m2 *sample_m2))

        fused_feature = torch.add(m1_embed.float(),m2_embed.float())
        fused_feature = self.cat_norm(fused_feature)

        sig_cat = model_cat(fused_feature)
        mean_cat, var_cat = sig_cat
        #mean_cat = torch.clamp(mean_cat, min=-5, max=5)
        var_cat = torch.clamp(var_cat, min=0, max=1)
        sample_cat = torch.normal(mean= torch.zeros(mean_cat.size()), std= torch.ones(mean_cat.size())).to(self.device)
        fused_feature = torch.add(fused_feature, self.softplus(mean_cat + var_cat *sample_cat))
        fused_feature = torch.relu(self.drop(self.pre_early_2(fused_feature))) ### [batch_size, 200]
        #fused_feature = torch.relu(self.drop(self.pre_early_3(fused_feature))) ### [batch_size, 200]
        clf = self.early(fused_feature)

        return clf

