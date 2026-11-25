import torch.nn as nn
import torch
import torch.nn.functional as F

class Meta_net_static(nn.Module):
    def __init__(self, dims, length = 20, bnn_p = 0.5):
        super(Meta_net_static, self).__init__()

        self.fc_m1 = nn.Linear(dims[0], length)
        self.fc_m2 = nn.Linear(dims[1], length)

    def forward(self, m1, m2):
        m1 = torch.relu(self.fc_m1(m1))
        m2 = torch.relu(self.fc_m2(m2))
        return [m1, m2]


class Meta_net(nn.Module):
    def __init__(self, dims, length = 20, bnn_p = 0.5):
        super(Meta_net, self).__init__()

        self.fc = nn.Linear(dims[0], length)
        self.fc_1 = nn.Linear(dims[0], length)
        self.fc_2 = nn.Linear(dims[1], length)
        self.fc_3 = nn.Linear(dims[1], length)

    def forward(self, im, aud):
        mu_im = torch.relu(self.fc(im))
        sigma_im = torch.relu(self.fc_1(im))

        mu_txt = torch.relu(self.fc_2(aud))
        sigma_txt = torch.relu(self.fc_3(aud))

        return [mu_im, sigma_im, mu_txt, sigma_txt]

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
    def __init__(self, isBnn,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3, num_f=4, num_sec = 2):
        super().__init__()
        self.cat_norm = nn.BatchNorm1d(num_f*length)

        self.im_dim = nn.Linear(49, 1, bias=False)

        im_dim = nn.Linear(512, num_f*length, bias=False)
        im_norm = nn.BatchNorm1d(num_f*length)
        #im_dim_1 = nn.Linear(4*length, 2*length, bias=False)
        #im_norm_1 = nn.BatchNorm1d(2*length)
        #im_dim_2= nn.Linear(2*length, length, bias=False)
        #im_norm_2 = nn.BatchNorm1d(length)
        self.im_list = nn.ModuleList([im_dim])
        self.im_norm_list = nn.ModuleList([im_norm])

        txt_dim = nn.Linear(768, num_f*length, bias=False)
        text_norm = nn.BatchNorm1d(num_f*length)
        #txt_dim_1 = nn.Linear(4*length, 2*length, bias=False)
        #text_norm_1 = nn.BatchNorm1d(2*length)
        #txt_dim_2 = nn.Linear(2*length, length, bias=False)
        #text_norm_2 = nn.BatchNorm1d(length)
        self.txt_list = nn.ModuleList([txt_dim])
        self.txt_norm_list = nn.ModuleList([text_norm])

        self.drop = nn.Dropout(p=functional_p)
        pre_early = nn.Linear(num_f*length, num_sec*length)
        pre_early_2 = nn.Linear(num_sec*length, length)
        pre_early_3 = nn.Linear(length, length)
        self.pre_early = nn.ModuleList([pre_early,pre_early_2,pre_early_3])
        self.early = nn.Linear(length, 3)
        self.softplus = torch.nn.Softplus(beta=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.isBnn = isBnn
    def forward(self,im, txt, model_list, model_cat_list):

        im_embed_0 = im.view(-1, 512, 49)
        im_embed_0 = torch.squeeze(torch.relu(self.drop(self.im_dim(im_embed_0))), 2) ###[batch_size, 512, 1]
        txt_embed_0 = torch.mean(txt,1) ###[batch_size, length, 768]

        for im_net, im_norm, txt_net, txt_norm, auxiliary, this_bnn in zip(self.im_list, self.im_norm_list, self.txt_list, self.txt_norm_list, model_list,self.isBnn):
            im_embed = torch.relu(im_net(im_embed_0)) ###[batch_size, 512, 1]
            im_embed = im_norm(im_embed) ###[batch_size, 512, 1]
            txt_embed = torch.relu(txt_net(txt_embed_0))
            txt_embed = txt_norm(txt_embed)

            if this_bnn:
               mean_im, var_im, mean_txt, var_txt = auxiliary(im_embed_0.clone(), txt_embed_0.clone())

               im_embed = self.combine(mean_im, var_im, im_embed)
               txt_embed = self.combine(mean_txt, var_txt, txt_embed)
            else:
               bnn_im, bnn_txt = auxiliary(im_embed_0.clone(), txt_embed_0.clone())
               im_embed = bnn_im + im_embed
               txt_embed = bnn_txt + txt_embed

            im_embed_0 = im_embed
            txt_embed_0 = txt_embed

        fused_feature = torch.add(im_embed.float(),txt_embed.float())
        fused_feature = self.cat_norm(fused_feature)

        for model_cat, pre_early in zip(model_cat_list, self.pre_early):
            mean_cat, var_cat = model_cat(fused_feature)
            fused_feature = self.combine(mean_cat, var_cat, fused_feature)
            fused_feature = torch.relu(self.drop(pre_early(fused_feature))) ### [batch_size, 200]

        clf = self.early(fused_feature)

        return clf

    def combine(self, mean, var, input):
        mean = torch.clamp(mean, min=-2, max=2)
        var = torch.clamp(var, min=0, max=2)
        sample = torch.normal(mean= torch.zeros(mean.size()), std= torch.ones(mean.size())).to(self.device)
        embed = torch.add(input, self.softplus(mean + var*sample ))  ###[batch_size, 512]

        return embed

