import torch.nn as nn
import torch
#from transformers import AutoImageProcessor, ViTModel
#from datasets import load_dataset, Dataset,Image

class Roberta(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206):
        super().__init__()
        self.drop = nn.Dropout(p=0.1)
    def forward(self,im, txt):
        txt_feature = torch.mean(txt, dim=1)
        return txt_feature

#class ViT(nn.Module):
#    def __init__(self,
#                 num_features_im=49, num_features_txt=768,
#                 im_dim = 197, txt_dim=206):
#        super().__init__()
#        print("This is visual transfomer model")
#        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
#        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
#        self.drop = nn.Dropout(p=0.1)
#        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
#        for param in self.model.parameters():
#            param.requires_grad = False

#    def forward(self,im, txt):
#
#        with torch.no_grad():
#             dataset = Dataset.from_dict({"image": ['.'+m for m in im]}).cast_column("image", Image())
#             image = dataset["image"]
#             inputs = self.image_processor(image, return_tensors="pt").to(self.device)
#             outputs = self.model(**inputs)
#             features = torch.mean(outputs.last_hidden_state,dim=1)
        
#        return features


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


class Early_fusion(nn.Module):
    def __init__(self,isBnn,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 197, txt_dim=206,
                 length = 20, dim=10, functional_p = 0.3, num_f=4, num_sec = 2):
        super().__init__()
        #self.im = ViT()
        self.txt = Roberta(num_features_im, num_features_txt,
                 im_dim, txt_dim)
        self.cat_norm = nn.BatchNorm1d(num_f*length)
        self.drop = nn.Dropout(p=functional_p)
        #self.im_dim = nn.Linear(49, 1, bias=False)

        im_dim = nn.Linear(768, num_f*length, bias=False)
        im_norm = nn.BatchNorm1d(num_f*length)
        self.im_list = nn.ModuleList([im_dim])
        self.im_norm_list = nn.ModuleList([im_norm])

        txt_dim = nn.Linear(768, num_f*length, bias=False)
        text_norm = nn.BatchNorm1d(num_f*length)
        self.txt_list = nn.ModuleList([txt_dim])
        self.txt_norm_list = nn.ModuleList([text_norm])

        self.drop = nn.Dropout(p=functional_p)
        pre_early = nn.Linear(num_f*length, length)
        #pre_early_2 = nn.Linear(num_sec*length, length)
        self.mu = nn.ModuleList([pre_early])
        
        pre_early = nn.Linear(num_f*length, length)
        #pre_early_2 = nn.Linear(num_sec*length, length)
        self.sigma = nn.ModuleList([pre_early])

        #self.early_0 = nn.Linear(length, length)
        self.early = nn.Linear(length, 5)
        self.softplus = torch.nn.Softplus(beta=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.isBnn = isBnn


    def forward(self,im, txt, model_list, model_cat_list):
        #with torch.no_grad():
        #im_embed_0 = self.im(im,txt)
        im_embed_0 = im
        txt_embed_0 = self.txt(im,txt)
        
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

        for model_cat, mu, sigma in zip(model_cat_list, self.mu, self.sigma):
            mean_cat, var_cat = model_cat(fused_feature)
            fused_feature = self.combine(mean_cat, var_cat, fused_feature)
            mean = torch.relu(self.drop(mu(fused_feature))) ### [batch_size, 200]
            var =  torch.relu(self.drop(sigma(fused_feature)))
            fused_feature = self.combine(mean, var, 0)

        #fused_feature = torch.relu(self.early_0(fused_feature))
        clf = self.early(fused_feature)

        return clf

    def combine(self, mean, var, input):
        #mean = torch.clamp(mean, min=-2, max=2)
        var = torch.clamp(var, min=-2, max=2)
        sample = torch.normal(mean= torch.zeros(mean.size()), std= torch.ones(mean.size())).to(self.device)
        embed = torch.add(input, self.softplus(mean + var*sample ))  ###[batch_size, 512]

        return embed


