import torch.nn as nn
import torch
import torch.nn.functional as F
from models import *
class EarlyFusionModel(nn.Module):
    def __init__(self, length = 20, out_dim=2, functional_p = 0.3):
        super().__init__()

        self.im_dim_1 = nn.Linear(768, length, bias=False)
        self.im_dim_2 = nn.Linear(length, out_dim, bias=False)
        self.text_dim_1 = nn.Linear(768, length, bias=False)
        self.text_dim_2 = nn.Linear(length, out_dim, bias=False)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.im = ViT().to(self.device)
        self.txt = Roberta(txt_dim=40).to(self.device)

        self.p = nn.Dropout(p=functional_p)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self,im, txt):

        im_embed = self.im(im,txt)[1]
        txt_embed = self.txt(im,txt)[1]

        im_embed= self.p(self.relu(self.im_dim_1(im_embed)))
        txt_embed = self.p(self.relu(self.text_dim_1(txt_embed)))

        im_embed = self.im_dim_2(im_embed)
        txt_embed = self.text_dim_2(txt_embed)
        return [im_embed+txt_embed, self.softmax(im_embed), self.softmax(txt_embed)]

