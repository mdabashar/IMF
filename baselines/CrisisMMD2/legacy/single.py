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


class Early_fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = nn.Linear(768, 5)
    def forward(self,im, txt):
        clf = self.early(im)
        return clf

