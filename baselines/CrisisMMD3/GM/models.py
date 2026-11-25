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
        self.text_dim = nn.Linear(num_features_txt, 1, bias=False)
        self.text_clf = nn.Linear(num_features_txt, 3, bias=False)
        self.drop = nn.Dropout(p=0.1)
        #self.txt_dim = txt_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self,im, txt):
        #txt_feature = torch.relu(self.drop(self.text_dim(txt)))
        #txt = torch.reshape(txt, (-1, self.txt_dim,768))
        txt = torch.tensor(txt,dtype=torch.float32).to(self.device)
        txt_feature = torch.mean(txt, dim =1)
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
        self.im_dim = nn.Linear(num_features_txt, 1, bias=False)
        self.im_clf = nn.Linear(num_features_txt, 3, bias=False)
        self.drop = nn.Dropout(p=0.1)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self,im, txt):

        dataset = Dataset.from_dict({"image": ['.'+m for m in im]}).cast_column("image", Image())
        image = dataset["image"]
        inputs = self.image_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        feature = torch.mean(outputs.last_hidden_state, dim=1)
        #feature = torch.relu(self.drop(self.im_dim(features)))

        im_clf = self.im_clf(feature)

        return [im_clf, feature]

class Intermediate(nn.Module):
    def __init__(self,length=100,functional_p=0.3,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 197, txt_dim=206):
        super().__init__()
        print("This is cross model")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.im = ViT().to(self.device)
        self.txt = Roberta(num_features_im, num_features_txt,
                 im_dim, txt_dim).to(self.device)

        self.im_1 = nn.Linear(num_features_txt, length)
        self.txt_1 = nn.Linear(num_features_txt, length)
        self.early = nn.Linear(length, 3)
        self.relu = torch.nn.ReLU()
        self.p = nn.Dropout(p=functional_p)

    def forward(self,im, txt, ratios):
        im_embed = self.im(im,txt)[1]
        txt_embed = self.txt(im,txt)[1]

        im_embed_1= self.p(self.relu(self.im_1(txt_embed)))
        txt_embed_1 = self.p(self.relu(self.txt_1(im_embed)))

        clf = self.early(im_embed_1*ratios[0]+txt_embed_1*ratios[1])
        
        return [clf, im_embed, txt_embed]



class RMFE(nn.Module):
    def __init__(self,length=100,functional_p=0.3,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 197, txt_dim=206):
        super().__init__()
        print("This is inter-rmfe model")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.im = ViT().to(self.device)
        self.txt = Roberta(num_features_im, num_features_txt,
                 im_dim, txt_dim).to(self.device)

        self.im_1 = nn.Linear(num_features_txt, length)
        self.txt_1 = nn.Linear(num_features_txt, length)
        self.early = nn.Linear(length, 3)
        self.relu = torch.nn.ReLU()
        self.p = nn.Dropout(p=functional_p)

    def forward(self,im, txt, augment=False):
        if not augment:
           im_embed = self.im(im,txt)[1]
           txt_embed = self.txt(im,txt)[1]
        else:
           im_embed = im
           txt_embed = txt

        im_embed_1= self.p(self.relu(self.im_1(txt_embed)))
        txt_embed_1 = self.p(self.relu(self.txt_1(im_embed)))

        clf = self.early(im_embed_1*0.5+txt_embed_1*0.5)
        
        return [clf, im_embed, txt_embed]
