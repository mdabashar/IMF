import torch.nn as nn
import torch
import torch.nn.functional as F

class Meta_net(nn.Module):
    def __init__(self, input_length, length = 100, bnn_p = 0.5):
        super(Meta_net, self).__init__()
        self.dropout = nn.Dropout(p=bnn_p)
        self.fc_1 = nn.Linear(input_length, input_length)
        self.fc_3 = nn.Linear(input_length, input_length)

    def forward(self, x):
        
        mu = torch.relu(self.dropout(self.fc_1(x))) ###[batch_size, 512, 1]
        sigma = torch.relu(self.dropout(self.fc_3(x))) ###[batch_size, 512, 1]
        return [mu, sigma]

class Early_fusion_new_0(nn.Module):
    def __init__(self, dim=200, length = 200, functional_p = 0.3):
        super().__init__()

        self.modality1 = nn.Linear(1000, length)
        self.modality2 = nn.Linear(1000, length)
        self.modality3 = nn.Linear(503, length)
        self.drop = nn.Dropout(p=functional_p)
        self.pre_early_2 = nn.Linear(length, length)
        self.early = nn.Linear(length, 5)
        self.softplus = torch.nn.Softplus(beta=1)
        #self.sigmoid = torch.nn.Sigmoid()

    def forward(self, modal_data_1, modal_data_2, modal_data_3,
                sigma_1, sigma_2, sigma_3,  model_cat):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
        #modality1_embed = torch.relu(self.drop(self.modality1(modal_data_1)))
        mean_1, var_1 = sigma_1
        #mean_1 = torch.clamp(mean_1, min=-2, max=2)
        var_1 = torch.clamp(var_1, min=0, max=1)
        sample_1 = torch.normal(mean= torch.zeros(mean_1.size()), std= torch.ones(mean_1.size())).to(device)
        modality1_embed = modal_data_1+self.softplus(mean_1 + var_1*sample_1 )
        modality1_embed = torch.relu(self.drop(self.modality1(modality1_embed)))

        #modality2_embed = torch.relu(self.drop(self.modality2(modal_data_2)))
        mean_2, var_2 = sigma_2
        #mean_2 = torch.clamp(mean_2, min=-2, max=2)
        var_2 = torch.clamp(var_2, min=0, max=1)
        sample_2 = torch.normal(mean= torch.zeros(mean_2.size()), std= torch.ones(mean_2.size())).to(device)
        modality2_embed = modal_data_2+self.softplus(mean_2 + var_2*sample_2 )       
        modality2_embed = torch.relu(self.drop(self.modality2(modality2_embed)))

        #modality3_embed = torch.relu(self.drop(self.modality3(modal_data_3)))
        mean_3, var_3 = sigma_3
        #mean_3 = torch.clamp(mean_3, min=-2, max=2)
        var_3 = torch.clamp(var_3, min=0, max=1)
        sample_3 = torch.normal(mean= torch.zeros(mean_3.size()), std= torch.ones(mean_3.size())).to(device)
        modality3_embed = modal_data_3+self.softplus(mean_3 + var_3*sample_3 )
        modality3_embed = torch.relu(self.drop(self.modality3(modality3_embed)))


        fused_feature = torch.add(modality2_embed.float(),modality3_embed.float())
        fused_feature = torch.add(fused_feature,modality1_embed.float())
        sig_cat = model_cat(fused_feature)
        mean_cat, var_cat = sig_cat
        #mean_cat = torch.clamp(mean_cat, min=-2, max=2)
        var_cat = torch.clamp(var_cat, min=0, max=1)
        sample_cat = torch.normal(mean= torch.zeros(mean_cat.size()), std= torch.ones(mean_cat.size())).to(device)
        fused_feature = fused_feature+self.softplus(mean_cat + var_cat *sample_cat)
        fused_feature = torch.relu(self.drop(self.pre_early_2(fused_feature)))
        clf = self.early(fused_feature)

        return clf

