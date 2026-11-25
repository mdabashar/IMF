import torch.nn as nn
import torch
import torch.nn.functional as F


class Early_fusion(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()
        self.im_dim = nn.Linear(49, 1, bias=False)
        self.drop = nn.Dropout(p=functional_p)
        self.early = nn.Linear(512+768, 3)

    def forward(self,im, txt, sigma_txt=0, sigma_im=0):
        im = im.view(-1, 512, 49)
        im_embed = torch.squeeze(torch.relu(self.drop(self.im_dim(im))), 2) ###[batch_size, 512, 1]
        txt_embed_0 = torch.mean(txt, 1) ###[batch_size, length, 768]
        cat_feature = torch.cat([im_embed.float(),txt_embed_0.float()], dim = 1)
        clf = self.early(cat_feature)
        return clf




class Early_fusion2(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()
        self.im_dim = nn.Linear(49, 1, bias=False)
        self.im_dim_1 = nn.Linear(512, length, bias=False)
        self.text_dim = nn.Linear(768, length, bias=False)
        self.drop = nn.Dropout(p=functional_p)
        self.pre_early_2 = nn.Linear(length, length)
        self.early = nn.Linear(length, 3)

    def forward(self,im, txt, sigma_txt=0, sigma_im=0):
        im = im.view(-1, 512, 49)
        im_embed = torch.squeeze(torch.relu(self.drop(self.im_dim(im))), 2) ###[batch_size, 512, 1]
        im_embed = torch.relu(self.drop(self.im_dim_1(im_embed))) ###[batch_size, 512, 1]
        txt_embed_0 = torch.mean(txt, 1) ###[batch_size, length, 768]
        txt_embed = torch.relu(self.drop(self.text_dim(txt_embed_0)))
        cat_feature = torch.add(im_embed.float(),txt_embed.float())
        cat_feature = torch.relu(self.drop(self.pre_early_2(cat_feature))) ### [batch_size, 200]
        clf = self.early(cat_feature)
        return clf





class Late_fusion(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()
        self.im_dim = nn.Linear(49, 1, bias=False)
        self.im_dim_1 = nn.Linear(512, length, bias=False)
        self.text_dim = nn.Linear(768, length, bias=False)
        #self.im_dim_2 = nn.Linear(length, int(length/2), bias=False)
        #self.text_dim_2 = nn.Linear(length, int(length/2), bias=False)

        # self.im_dim_3 = nn.Linear(int(length/2), int(length/4))
        # self.text_dim_3 = nn.Linear(int(length/2), int(length/4))

        self.im_dim_4 = nn.Linear(length, 3)
        self.text_dim_4 = nn.Linear(length, 3)

        self.drop = nn.Dropout(p=functional_p)

    def forward(self,im, txt, sigma_txt=0, sigma_im=0):

        # im_embed = torch.squeeze(im,1)
        im = im.view(-1, 512, 49)
        im_embed = torch.squeeze(torch.relu(self.drop(self.im_dim(im))), 2) ###[batch_size, 512, 1]
        im_embed = torch.relu(self.drop(self.im_dim_1(im_embed))) ###[batch_size, 512, 1]
        #im_embed = torch.relu(self.drop(self.im_dim_2(im_embed))) ###[batch_size, 512, 1]
        # im_embed = torch.relu(self.drop(self.im_dim_3(im_embed))) ###[batch_size, 512, 1]
        im_embed = torch.relu(self.drop(self.im_dim_4(im_embed))) ###[batch_size, 512, 1]

        txt_embed = torch.mean(txt,1) ###[batch_size, length, 768]
        txt_embed = torch.relu(self.drop(self.text_dim(txt_embed)))
        #txt_embed = torch.relu(self.drop(self.text_dim_2(txt_embed)))
        # txt_embed = torch.relu(self.drop(self.text_dim_3(txt_embed)))
        txt_embed = torch.relu(self.drop(self.text_dim_4(txt_embed)))

        clf = torch.add(im_embed, txt_embed)/2
        return clf

class Cross(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()

        self.im_dim = nn.Linear(49, 1, bias=False)
        self.im_dim_1 = nn.Linear(512, length, bias=False)
        self.text_dim = nn.Linear(768, length, bias=False)

        self.im_sigma = nn.Linear(length, length, bias=False)
        self.text_sigma = nn.Linear(length, length, bias=False)
        self.drop = nn.Dropout(p=functional_p)
        self.pre_early = nn.Linear(length, length)
        self.pre_early_2 = nn.Linear(length, int(length/2))
        self.pre_early_3 = nn.Linear(int(length/2), int(length/4))

        self.early = nn.Linear(int(length/4), 3)
    def forward(self,im, txt, sigma_txt=0, sigma_im=0):

        # im_embed = torch.squeeze(im,1)
        im_embed_0 = im.view(-1, 512, 49)
        im_embed = torch.squeeze(torch.relu(self.drop(self.im_dim(im_embed_0))), 2) ###[batch_size, 512, 1]
        im_embed = torch.relu(self.drop(self.im_dim_1(im_embed))) ###[batch_size, 512, 1]

        txt_embed = torch.mean(txt,1) ###[batch_size, length, 768]
        txt_embed = torch.relu(self.drop(self.text_dim(txt_embed)))

        sigma_txt = torch.sigmoid(self.drop(self.text_sigma(im_embed))) ###[batch_size, 512, 1]
        sigma_im = torch.sigmoid(self.drop(self.im_sigma(txt_embed))) ###[batch_size, 512, 1]

        im_embed = im_embed*sigma_im
        txt_embed = txt_embed*sigma_txt

        cat_feature = torch.add(im_embed.float(),txt_embed.float())
        cat_feature = torch.relu(self.drop(self.pre_early(cat_feature))) ### [batch_size, 200]
        fused_feature = torch.relu(self.drop(self.pre_early_2(cat_feature))) ### [batch_size, 200]
        cat_feature = torch.relu(self.drop(self.pre_early_3(fused_feature))) ### [batch_size, 200]
        clf = self.early(cat_feature)
        return clf



class GMU(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()

        self.im_dim = nn.Linear(49, 1, bias=False)
        self.im_dim_1 = nn.Linear(512, length, bias=False)
        self.text_dim = nn.Linear(768, length, bias=False)

        self.sigma = nn.Linear(2*length, length, bias=False)
        self.drop = nn.Dropout(p=functional_p)
        self.pre_early = nn.Linear(length, length)
        self.pre_early_2 = nn.Linear(length, length)
        self.pre_early_3 = nn.Linear(length, length)

        self.early = nn.Linear(length, 3)
    def forward(self,im, txt, sigma_txt=0, sigma_im=0):

        # im_embed = torch.squeeze(im,1)
        im_embed_0 = im.view(-1, 512, 49)
        im_embed = torch.squeeze(torch.relu(self.drop(self.im_dim(im_embed_0))), 2) ###[batch_size, 512, 1]
        im_embed = torch.tanh(self.drop(self.im_dim_1(im_embed))) ###[batch_size, 512, 1]

        txt_embed = torch.mean(txt,1) ###[batch_size, length, 768]
        txt_embed = torch.tanh(self.drop(self.text_dim(txt_embed)))

        weight = torch.sigmoid(self.sigma(torch.cat([txt_embed, im_embed], dim=1)))

        cat_feature = torch.add(im_embed.float()* weight,txt_embed.float()*(1-weight))
        cat_feature = torch.relu(self.drop(self.pre_early(cat_feature))) ### [batch_size, 200]
        fused_feature = torch.relu(self.drop(self.pre_early_2(cat_feature))) ### [batch_size, 200]
        cat_feature = torch.relu(self.drop(self.pre_early_3(fused_feature))) ### [batch_size, 200]
        clf = self.early(cat_feature)
        return clf
class Tensor(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()

        self.im_dim = nn.Linear(49, 1, bias=False)
        self.im_dim_1 = nn.Linear(512, length, bias=False)
        self.text_dim = nn.Linear(768, length, bias=False)

        self.drop = nn.Dropout(p=functional_p)

        self.early_1 = nn.Linear((length+1)*(length+1), length+1)
        self.early = nn.Linear(length+1, 3)
    def forward(self,im, txt):

        im_embed_0 = im.view(-1, 512, 49)
        im_embed = torch.squeeze(torch.relu(self.drop(self.im_dim(im_embed_0))), 2) ###[batch_size, 512, 1]
        txt_embed = torch.mean(txt,1) ###[batch_size, length, 768]

        im_embed = torch.relu(self.drop(self.im_dim_1(im_embed)))
        txt_embed = torch.relu(self.drop(self.text_dim(txt_embed)))

        im_embed = torch.cat([im_embed, torch.ones(im_embed.size()[0],1)], dim=1)
        txt_embed = torch.cat([txt_embed,  torch.ones(im_embed.size()[0],1)], dim=1)

        cat_feature = torch.bmm(im_embed.unsqueeze(2),txt_embed.unsqueeze(1)).view(im_embed.size()[0], -1)
        cat_feature = torch.relu(self.early_1(cat_feature))
        clf = self.early(cat_feature)
        return clf

class TensorGate(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768, gate="im",
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()

        self.im_dim = nn.Linear(49, 1, bias=False)
        self.im_dim_1 = nn.Linear(512, length, bias=False)
        self.text_dim = nn.Linear(768, length, bias=False)
        self.gate = gate
        self.drop = nn.Dropout(p=functional_p)

        self.early_1 = nn.Linear((length+1)*(length+1), length+1)
        self.early = nn.Linear(length+1, 3)
    def forward(self,im, txt):
        im_embed_0 = im.view(-1, 512, 49)
        im_embed = torch.squeeze(torch.relu(self.drop(self.im_dim(im_embed_0))), 2) ###[batch_size, 512, 1]
        txt_embed = torch.mean(txt,1) ###[batch_size, length, 768]

        im_embed = self.drop(self.im_dim_1(im_embed))
        txt_embed = self.drop(self.text_dim(txt_embed))

        if self.gate == "im":
           im_embed = torch.sigmoid(im_embed)
           txt_embed = torch.relu(txt_embed)
        else:
           im_embed = torch.relu(im_embed)
           txt_embed = torch.sigmoid(txt_embed)

        im_embed = torch.cat([im_embed, torch.ones(im_embed.size()[0],1)], dim=1)
        txt_embed = torch.cat([txt_embed,  torch.ones(im_embed.size()[0],1)], dim=1)
        cat_feature = torch.bmm(im_embed.unsqueeze(2),txt_embed.unsqueeze(1)).view(im_embed.size()[0], -1)
        cat_feature = torch.relu(self.early_1(cat_feature))
        clf = self.early(cat_feature)
        return clf


class Text(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()

        self.text_dim = nn.Linear(768, length, bias=False)

        #self.pre_early_2 = nn.Linear(length, int(length/2))
        #self.pre_early_3 = nn.Linear(int(length/2), int(length/4))
        # self.pre_early_1 = nn.Linear(dim, 1)
        # self.early = nn.Linear(int(length/4), 2)
        self.early = nn.Linear(length, 3)
        self.drop = nn.Dropout(p=functional_p)

    def forward(self, txt, sigma_txt=0, sigma_im=0):

        txt_embed_0 = torch.mean(txt,1) ###[batch_size, length, 768]
        txt_embed = torch.relu(self.drop(self.text_dim(txt_embed_0)))
        #txt_embed = self.drop(self.pre_early_2(txt_embed))
        #txt_embed = self.drop(self.pre_early_3(txt_embed))
        txt_embed = self.drop(self.early(txt_embed))

        return txt_embed


class Image(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()

        self.im_dim = nn.Linear(49, 1, bias=False)
        self.im_dim_1 = nn.Linear(512, length, bias=False)
        #self.pre_early_2 = nn.Linear(length, int(length/2))
        #self.pre_early_3 = nn.Linear(int(length/2), int(length/4))
        # self.pre_early_1 = nn.Linear(dim, 1)
        # self.early = nn.Linear(int(length/4), 2)
        self.early = nn.Linear(length, 3)
        self.drop = nn.Dropout(p=functional_p)

    def forward(self,im, sigma_txt=0, sigma_im=0):

        # im_embed_0 = torch.squeeze(im,1)
        # im_embed = torch.relu(self.drop(self.im_dim(im_embed_0))) ###[batch_size, 512, 1]

        im_embed_0 = im.view(-1, 512, 49)
        im_embed = torch.squeeze(torch.relu(self.drop(self.im_dim(im_embed_0))), 2) ###[batch_size, 512, 1]
        im_embed = torch.relu(self.drop(self.im_dim_1(im_embed))) ###[batch_size, 512, 1]

        #im_embed = self.drop(self.pre_early_2(im_embed)) ###[batch_size, 512, 1]
        #im_embed = self.drop(self.pre_early_3(im_embed))
        im_embed = self.drop(self.early(im_embed))

        return im_embed
