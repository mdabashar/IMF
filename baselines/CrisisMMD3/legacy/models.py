import torch.nn as nn
import torch
import torch.nn.functional as F

def uncertaintyNormalization(uncertainty_im, uncertainty_txt):
    n_im = (1- (uncertainty_im + (1e-15))/(uncertainty_im + uncertainty_txt + 2*(1e-15)))
    n_txt = (1- (uncertainty_txt + (1e-15))/(uncertainty_im + uncertainty_txt + 2*(1e-15)))
    # print("BEFORE: ", uncertainty_im)
    # print("AFTER: ", n_im)
    return n_im, n_txt


class Meta_net(nn.Module):
    def __init__(self, input_dim, input_length, dim = 10,
                 isImage=False, length = 100, bnn_p = 0.5, other_dim = 40):
        super(Meta_net, self).__init__()
        self.dropout = nn.Dropout(p=bnn_p)
        self.fc = nn.Linear(49, 1)
        self.fc_1 = nn.Linear(512, length)
        # self.norm = nn.BatchNorm1d(length)

        self.fc_2 = nn.Linear(49, 1)
        self.fc_3 = nn.Linear(512, length)
        # self.norm2 = nn.BatchNorm1d(length)

        self.isImage = isImage


    def forward(self, x):


        # x = torch.squeeze(x,1)
        x = x.view(-1, 512, 49)
        mu = torch.squeeze(self.dropout(torch.relu(self.fc(x))), 2) ###[batch_size, 512, 1]
        mu = torch.relu(self.dropout(self.fc_1(mu))) ###[batch_size, 512, 1]

        sigma = torch.squeeze(torch.relu(self.dropout(self.fc_2(x))), 2) ###[batch_size, 512, 1]
        sigma = torch.relu(self.dropout(self.fc_3(sigma))) ###[batch_size, 512, 1]
        return [mu, sigma]



class Meta_net(nn.Module):
    def __init__(self, input_length, dims, length = 20, bnn_p = 0.5):
        super(Meta_net, self).__init__()

        self.fc = nn.Linear(dims[0], length)
        self.fc_1 = nn.Linear(dims[0], length)

        self.fc_2 = nn.Linear(dims[1], length)
        self.fc_3 = nn.Linear(dims[1], length)

    def forward(self, im, aud):
        mu_im = torch.relu(self.fc(im))
        sigma_im = torch.relu(self.fc_1(im))

        mu_aud = torch.relu(self.fc_2(aud))
        sigma_aud = torch.relu(self.fc_3(aud))

        return [mu_im, sigma_im, mu_aud, sigma_aud]


class Meta_net_cat(nn.Module):
    def __init__(self, length = 20, bnn_p = 0.5, txt_dim=40, im_dim=512):
        super(Meta_net_cat, self).__init__()
        self.dropout = nn.Dropout(p=bnn_p)
        self.fc = nn.Linear(length, length)
        self.fc_2 = nn.Linear(length, length)

    def forward(self, x):
        mu = torch.relu(self.dropout(self.fc(x)))
        sigma = torch.relu(self.dropout(self.fc_2(x)))
        return [mu, sigma]


class Early_fusion_new_0(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()

        self.im_norm = nn.BatchNorm1d(length)
        self.text_norm = nn.BatchNorm1d(length)
        self.cat_norm = nn.BatchNorm1d(length)
        self.im_dim_1 = nn.Linear(512, length, bias=False)
        self.im_dim = nn.Linear(49, 1, bias=False)
        self.text_dim = nn.Linear(768, length, bias=False)
        # self.im_weight = nn.Linear(length, length, bias=False)
        # self.text_weight = nn.Linear(length, length, bias=False)
        self.drop = nn.Dropout(p=functional_p)
        self.pre_early_2 = nn.Linear(length, length)
        #self.pre_early_3 = nn.Linear(length, int(length/2))
        self.early = nn.Linear(length, 3)
        self.softplus = torch.nn.Softplus(beta=1)

    def forward(self,im, txt, sigma_txt, sigma_im, model_cat):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # im = torch.squeeze(im,1)

        im_embed_0 = im.view(-1, 512, 49)
        im_embed_0 = torch.squeeze(torch.relu(self.drop(self.im_dim(im_embed_0))), 2) ###[batch_size, 512, 1]
        mean_im, var_im = sigma_im
        #mean_im = torch.clamp(mean_im, min=-5, max=5)
        var_im = torch.clamp(var_im, min=0, max=1)
        sample_im = torch.normal(mean= torch.zeros(mean_im.size()), std= torch.ones(mean_im.size())).to(device)
        
        im_embed = torch.relu(self.drop(self.im_dim_1(im_embed_0))) ###[batch_size, 512, 1]
        im_embed = self.im_norm(im_embed)
        im_embed = torch.add(im_embed, self.softplus(mean_im + var_im*sample_im ))  ###[batch_size, 512]

        txt_embed_0 = torch.mean(txt,1) ###[batch_size, length, 768]
        mean_txt, var_txt = sigma_txt
        #mean_txt = torch.clamp(mean_txt, min=-5, max=5)
        var_txt = torch.clamp(var_txt, min=0, max=1)
        sample_txt = torch.normal(mean= torch.zeros(mean_txt.size()), std= torch.ones(mean_txt.size())).to(device)
        
        txt_embed = torch.relu(self.drop(self.text_dim(txt_embed_0))) ###[batch_size, 512, 1]
        txt_embed = self.text_norm(txt_embed)
        txt_embed = torch.add(txt_embed, self.softplus(mean_txt + var_txt *sample_txt ))

        fused_feature = torch.add(im_embed.float(),txt_embed.float())
        fused_feature = self.cat_norm(fused_feature)

        sig_cat = model_cat(fused_feature)
        mean_cat, var_cat = sig_cat
        #mean_cat = torch.clamp(mean_cat, min=-5, max=5)
        var_cat = torch.clamp(var_cat, min=0, max=1)
        sample_cat = torch.normal(mean= torch.zeros(mean_cat.size()), std= torch.ones(mean_cat.size())).to(device)
        fused_feature = torch.add(fused_feature, self.softplus(mean_cat + var_cat *sample_cat))
        fused_feature = torch.relu(self.drop(self.pre_early_2(fused_feature))) ### [batch_size, 200]
        #fused_feature = torch.relu(self.drop(self.pre_early_3(fused_feature))) ### [batch_size, 200]
        clf = self.early(fused_feature)

        return clf

