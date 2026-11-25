import torch.nn as nn
import torch
from transformers import AutoImageProcessor, ViTModel
from datasets import load_dataset, Dataset,Image

class Roberta(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206):
        super().__init__()
        print("This is roberta model")
        self.text_clf = nn.Linear(num_features_txt, 1, bias=False)
        self.drop = nn.Dropout(p=0.1)
    def forward(self,im, txt):
        txt_feature = torch.mean(txt, dim=1)
        txt_clf = self.text_clf(txt_feature)
        return [txt_clf, txt_feature]

class ViT(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 197, txt_dim=206):
        super().__init__()
        print("This is visual transfomer model")
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.im_clf = nn.Linear(num_features_txt, 1, bias=False)
        self.drop = nn.Dropout(p=0.1)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    def forward(self,im, txt):

        dataset = Dataset.from_dict({"image": ['.'+m for m in im]}).cast_column("image", Image())
        image = [image.convert('RGB') for image in dataset["image"]]
        #image = dataset["image"]
        inputs = self.image_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        features = torch.mean(outputs.last_hidden_state,dim=1)
        
        im_clf = self.im_clf(features)

        return [im_clf, features]

class Early_fusion(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 197, txt_dim=206):
        super().__init__()
        print("This is early model")
        self.im = ViT()
        self.txt = Roberta(num_features_im, num_features_txt,
                 im_dim, txt_dim)
        self.early = nn.Linear(2*num_features_txt, 1)

    def forward(self,im, txt):
        im_embed = self.im(im,txt)[1]
        txt_embed = self.txt(im,txt)[1]
        clf = self.early(torch.cat([im_embed,txt_embed],dim=1))
        return [clf]

class Tensor(nn.Module):
    def __init__(self, length,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 197, txt_dim=206):
        super().__init__()
        print("This is early model")
        self.im = ViT()
        self.txt = Roberta(num_features_im, num_features_txt,
                 im_dim, txt_dim)
        self.im_fc = nn.Linear(num_features_txt, length)
        self.txt_fc = nn.Linear(num_features_txt, length)
        self.early_1 = nn.Linear((length+1)*(length+1), length+1)
        self.early = nn.Linear(length+1, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self,im, txt):
        im_embed = self.im(im,txt)[1]
        txt_embed = self.txt(im,txt)[1]

        im_embed = torch.relu(self.im_fc(im_embed))
        txt_embed = torch.relu(self.txt_fc(txt_embed))
        
        im_embed = torch.cat([im_embed, torch.ones(im_embed.size()[0],1).to(self.device)], dim=1)
        txt_embed = torch.cat([txt_embed,  torch.ones(im_embed.size()[0],1).to(self.device)], dim=1)

        cat_feature = torch.bmm(im_embed.unsqueeze(2),txt_embed.unsqueeze(1)).view(im_embed.size()[0], -1)
        cat_feature = torch.relu(self.early_1(cat_feature))
        clf = self.early(cat_feature)
        return [clf]



class Tensorgate(nn.Module):
    def __init__(self, length,gate="txt",
                 num_features_im=49, num_features_txt=768,
                 im_dim = 197, txt_dim=206):
        super().__init__()
        print("This is early model")
        self.im = ViT()
        self.txt = Roberta(num_features_im, num_features_txt,
                 im_dim, txt_dim)
        self.im_fc = nn.Linear(num_features_txt, length)
        self.txt_fc = nn.Linear(num_features_txt, length)
        self.early_1 = nn.Linear((length+1)*(length+1), length+1)
        self.early = nn.Linear(length+1, 1)
        self.gate = gate
        self.gate_fc = nn.Linear(num_features_txt, length)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self,im, txt):
        im_embed = self.im(im,txt)[1]
        txt_embed = self.txt(im,txt)[1]
       
        if self.gate == "im":
           im_embed_0 = torch.relu(self.im_fc(im_embed))
           im_gate = torch.sigmoid(self.gate_fc(im_embed))
           im_embed = im_gate*im_embed_0
           txt_embed = torch.relu(self.txt_fc(txt_embed))
        else:
           im_embed = torch.relu(self.im_fc(im_embed))
           txt_embed_0 = torch.relu(self.txt_fc(txt_embed)) 
           txt_gate = torch.sigmoid(self.gate_fc(txt_embed))
           txt_embed = txt_embed_0*txt_gate

        im_embed = torch.cat([im_embed, torch.ones(im_embed.size()[0],1).to(self.device)], dim=1)
        txt_embed = torch.cat([txt_embed,  torch.ones(im_embed.size()[0],1).to(self.device)], dim=1)

        cat_feature = torch.bmm(im_embed.unsqueeze(2),txt_embed.unsqueeze(1)).view(im_embed.size()[0], -1)
        cat_feature = torch.relu(self.early_1(cat_feature))
        clf = self.early(cat_feature)
        return [clf]


class Late_fusion(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 197, txt_dim=206):
        super().__init__()
        print("This is late model")
        self.im = ViT()
        self.txt = Roberta(num_features_im, num_features_txt,
                 im_dim, txt_dim)

    def forward(self,im, txt):
        im_embed = self.im(im,txt)[0]
        txt_embed = self.txt(im,txt)[0]
        clf = torch.add(im_embed, txt_embed)/2
        # print(clf)
        return [clf]

class Cross(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 197, txt_dim=206):
        super().__init__()
        print("This is cross model")
        self.im = ViT()
        self.txt = Roberta(num_features_im, num_features_txt,
                 im_dim, txt_dim)

        self.txt_atten = nn.Linear(num_features_txt, num_features_txt)
        self.im_atten = nn.Linear(num_features_txt, num_features_txt)
        self.early = nn.Linear(2*num_features_txt, 1)


    def forward(self,im, txt):
        im_embed = self.im(im,txt)[1]
        txt_embed = self.txt(im,txt)[1]

        im_cross= torch.sigmoid(self.im_atten(txt_embed))
        txt_cross = torch.sigmoid(self.txt_atten(im_embed))

        im_embed = im_cross*im_embed
        txt_embed = txt_cross*txt_embed

        clf = self.early(torch.cat([im_embed,txt_embed],dim=1))
        return [clf]


class GMU(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 197, txt_dim=206):
        super().__init__()
        print("This is cross model")
        self.im = ViT()
        self.txt = Roberta(num_features_im, num_features_txt,
                 im_dim, txt_dim)

        self.atten = nn.Linear(2*num_features_txt, num_features_txt)
        self.early = nn.Linear(num_features_txt, 1)

    def forward(self,im, txt):
        im_embed = self.im(im,txt)[1]
        txt_embed = self.txt(im,txt)[1]

        gate= torch.sigmoid(self.atten(torch.cat([txt_embed,im_embed],dim=1)))
        
        im_embed = gate*torch.tanh(im_embed)
        txt_embed = (1-gate)*torch.tanh(txt_embed)

        clf = self.early(im_embed+txt_embed)
        return [clf]
