import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, ViTModel
from datasets import load_dataset, Dataset,Image

class Roberta(nn.Module):
        def __init__(self, num_features_im=49, num_features_txt=768,im_dim = 512, txt_dim=206):
            super().__init__()
            print("This is roberta model")
            self.text_clf = nn.Linear(num_features_txt, 1, bias=False)
            self.drop = nn.Dropout(p=0.1)
        def forward(self,im, txt):
            txt_feature = torch.mean(txt, dim=1)
            txt_clf = self.text_clf(txt_feature)
            return [txt_clf, txt_feature]

class ViT(nn.Module):
        def __init__(self, num_features_im=49, num_features_txt=768,im_dim = 197, txt_dim=206):
            super().__init__()
            print("This is visual transfomer model")
            self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            self.model = model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.im_clf = nn.Linear(num_features_txt, 1, bias=False)
            self.drop = nn.Dropout(p=0.1)
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        def forward(self,im, txt):

            dataset = Dataset.from_dict({"image": ['../.'+m for m in im]}).cast_column("image", Image())
            image = [image.convert('RGB') for image in dataset["image"]]
            inputs = self.image_processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            features = torch.mean(outputs.last_hidden_state,dim=1)
            im_clf = self.im_clf(features)
            return [im_clf, features]


class IB(nn.Module):
    def __init__(self, dim=200, length = 200, functional_p = 0.3, p_gamma = 8, p_beta = 32, p_lambda = 0.3):
        super().__init__()
        self.im = ViT()
        self.txt = Roberta()
        self.drop_prob = functional_p
        self.length = length
        self.p_gamma = p_gamma 
        self.p_beta = p_beta 
        self.p_lambda = p_lambda
        self.encoder1 = nn.Sequential(
             
            nn.Linear(768, length),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.length, self.length * 2),
        )
        self.MLP1 = nn.Sequential(
            nn.Linear(self.length, length),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.length, 768),
            nn.Sigmoid(),
            nn.Dropout(self.drop_prob),
        )
        
        self.criterion = nn.MSELoss()
        self.early = nn.Linear(length, 3)


    def forward(self, m1, m2):
        m1 = self.im(m1,m2)[1]
        m2 = self.txt(m1,m2)[1]
        h1 = self.encoder1(m1)
        mu1, logvar1 = h1.chunk(2, dim=-1)
        kl_loss_0 = self.kl_loss(mu1, logvar1)
        b0 = self.reparameterise(mu1, logvar1)
        output1 = self.MLP1(b0)
        mse_0 = self.criterion(output1, m2)
        IB0 = kl_loss_0 + self.p_beta * mse_0

        clf = self.early(b0)

        return clf, IB0, kl_loss_0, mse_0, 0, 0


    def kl_loss(self, mu, logvar):
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_mean = torch.mean(kl_div)
        return kl_mean

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

