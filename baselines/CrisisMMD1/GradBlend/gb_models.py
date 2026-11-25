import torch.nn as nn
import torch
from transformers import AutoImageProcessor, ViTModel
from datasets import load_dataset, Dataset, Image

class Roberta(nn.Module):
    def __init__(self,length,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206):
        super().__init__()
        print("This is roberta model")
        self.drop = nn.Dropout(p=0.1)
        self.txt_dim = txt_dim
        self.text_layer = nn.Linear(num_features_txt, length) 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self,im, txt):
        #txt_feature = torch.relu(self.drop(self.text_dim(txt)))
        #txt = torch.reshape(txt, (-1, self.txt_dim,768))
        txt = torch.tensor(txt,dtype=torch.float32).to(self.device)
        txt_feature = torch.mean(txt, dim =1)
        txt_feature = self.drop(torch.relu(self.text_layer(txt_feature)))
        return txt_feature

class ViT(nn.Module):
    def __init__(self,length,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 197, txt_dim=206):
        super().__init__()
        print("This is visual transfomer model")
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.drop = nn.Dropout(p=0.1)
        self.im_layer = nn.Linear(num_features_txt, length)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    def forward(self,im, txt):

        dataset = Dataset.from_dict({"image": ['.'+m for m in im]}).cast_column("image", Image())
        image = [image.convert('RGB') for image in dataset["image"]] 
        #image = dataset["image"]
        inputs = self.image_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        feature = torch.mean(outputs.last_hidden_state, dim=1)
        feature = self.drop(torch.relu(self.im_layer(feature)))
        return feature



class MLP(torch.nn.Module):
    """Two layered perceptron."""

    def __init__(self, indim, outdim, dropout=False, dropoutp=0.1):

        super(MLP, self).__init__()
        self.fc = nn.Linear(indim, outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.norm = torch.nn.BatchNorm1d(indim)
    def forward(self, x):
        x = self.norm(x)
        output = torch.relu(self.fc(x))
        if self.dropout:
            output = self.dropout_layer(output)
        return output

class Concat(nn.Module):
    """Concatenation of input data on dimension 1."""

    def __init__(self):
        """Initialize Concat Module."""
        super(Concat, self).__init__()

    def forward(self, modalities):
        """
        Forward Pass of Concat.

        :param modalities: An iterable of modalities to combine
        """
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)
