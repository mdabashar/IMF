import torch.nn as nn
import torch
import torch.nn.functional as F


class Meta_net_static(nn.Module):
    def __init__(self, dims, length = 20, bnn_p = 0.5):
        super(Meta_net_static, self).__init__()

        self.fc_m1 = nn.Linear(dims[0], length)
        self.fc_m2 = nn.Linear(dims[1], length)
        self.fc_m3 = nn.Linear(dims[2], length)

    def forward(self, m1, m2, m3):
        m1 = torch.relu(self.fc_m1(m1))
        m2 = torch.relu(self.fc_m2(m2))
        m3 = torch.relu(self.fc_m3(m3))
        return [m1, m2, m3]

class Meta_net(nn.Module):
    def __init__(self, dims, length = 20, bnn_p = 0.5):
        super(Meta_net, self).__init__()

        self.fc_m1_mu = nn.Linear(dims[0], length)
        self.fc_m1_var = nn.Linear(dims[0], length)

        self.fc_m2_mu = nn.Linear(dims[1], length)
        self.fc_m2_var = nn.Linear(dims[1], length)

        self.fc_m3_mu = nn.Linear(dims[2], length)
        self.fc_m3_var = nn.Linear(dims[2], length)


    def forward(self, m1, m2, m3):
        mu_m1 = torch.relu(self.fc_m1_mu(m1))
        var_m1 = torch.relu(self.fc_m1_var(m1))

        mu_m2 = torch.relu(self.fc_m2_mu(m2))
        var_m2 = torch.relu(self.fc_m2_var(m2))

        mu_m3 = torch.relu(self.fc_m3_mu(m3))
        var_m3 = torch.relu(self.fc_m3_var(m3))

        return [mu_m1, var_m1, mu_m2, var_m2, mu_m3, var_m3]

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

        m1_dim = nn.Linear(1000, num_f*length, bias=False)
        #m1_dim_1 = nn.Linear(num_f*length, num_sec*length, bias=False)
        #m1_dim_2= nn.Linear(num_sec*length, length, bias=False)
        self.m1_list = nn.ModuleList([m1_dim])

        m2_dim = nn.Linear(1000, num_f*length, bias=False)
        #m2_dim_1 = nn.Linear(num_f*length, num_sec*length, bias=False)
        #m2_dim_2= nn.Linear(num_sec*length, length, bias=False)
        self.m2_list = nn.ModuleList([m2_dim])

        m3_dim = nn.Linear(503, num_f*length, bias=False)
        #m3_dim_1 = nn.Linear(num_f*length, num_sec*length, bias=False)
        #m3_dim_2= nn.Linear(num_sec*length, length, bias=False)
        self.m3_list = nn.ModuleList([m3_dim])

        self.drop = nn.Dropout(p=functional_p)

        pre_early_1 = nn.Linear(num_f*length, num_sec*length)
        self.mu = nn.ModuleList([pre_early_1])
       
        pre_early_1 = nn.Linear(num_f*length,num_sec*length)
        self.sigma = nn.ModuleList([pre_early_1])

        pre_early_1 = nn.BatchNorm1d(num_sec*length)
        self.norm = nn.ModuleList([pre_early_1])

        self.early = nn.Linear(length, 5)
        self.softplus = torch.nn.Softplus(beta=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.isBnn = isBnn

    def forward(self,m1_0, m2_0, m3_0, model_list, model_cat_list):

        for m1_net, m2_net, m3_net, auxiliary, this_bnn in zip(self.m1_list, self.m2_list, self.m3_list, model_list, self.isBnn):
            m1 = torch.relu(m1_net(m1_0))
            m2 = torch.relu(m2_net(m2_0))
            m3 = torch.relu(m3_net(m3_0))

            if this_bnn:
               mu_m1, var_m1, mu_m2, var_m2, mu_m3, var_m3 = auxiliary(m1_0.clone(), m2_0.clone(), m3_0.clone())

               m1 = self.combine(mu_m1, var_m1, m1)
               m2 = self.combine(mu_m2, var_m2, m2)
               m3 = self.combine(mu_m3, var_m3, m3)
            else:
                bnn_m1, bnn_m2, bnn_m3 = auxiliary(m1_0.clone(), m2_0.clone(), m3_0.clone())
                m1 = m1 + bnn_m1
                m2 = m2 + bnn_m2
                m3 = m3 + bnn_m3

            m1_0 = m1
            m2_0 = m2
            m3_0 = m3

        fused_feature = torch.add(m1.float(), m2.float())
        fused_feature = torch.add(fused_feature.float(), m3.float())
        fused_feature = self.cat_norm(fused_feature)

        for model_cat, mu, sigma, norm in zip(model_cat_list, self.mu, self.sigma, self.norm):

            mean_cat, var_cat = model_cat(fused_feature)
            fused_feature = self.combine(mean_cat, var_cat, fused_feature)
            mean = torch.relu(self.drop(mu(fused_feature))) ### 
            var  = torch.relu(self.drop(sigma(fused_feature)))
            fused_feature = norm(self.combine(mean, var, 0))

        clf = self.early(fused_feature)

        return clf

    def combine(self, mean, var, input):
        mean = torch.clamp(mean, min=-1, max=1)
        var = torch.clamp(var, min=0, max=2)
        sample = torch.normal(mean= torch.zeros(mean.size()), std= torch.ones(mean.size())).to(self.device)
        embed = torch.add(input, self.softplus(mean + var*sample ))  ###[batch_size, 512]

        return embed

