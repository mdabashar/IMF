import torch.nn as nn
import torch

class Text(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()

        self.text_dim = nn.Linear(768, length, bias=False)
        self.drop = nn.Dropout(p=functional_p)

    def forward(self, txt, sigma_txt=0, sigma_im=0):

        txt_embed_0 = torch.mean(txt,1) ###[batch_size, length, 768]
        txt_embed = torch.relu(self.drop(self.text_dim(txt_embed_0)))
        return txt_embed


class Image(nn.Module):
    def __init__(self,
                 num_features_im=49, num_features_txt=768,
                 im_dim = 512, txt_dim=206, length = 20, dim=10, functional_p = 0.3):
        super().__init__()

        self.im_dim = nn.Linear(49, 1, bias=False)
        self.im_dim_1 = nn.Linear(512, length, bias=False)
        self.drop = nn.Dropout(p=functional_p)

    def forward(self,im, sigma_txt=0, sigma_im=0):

        im_embed_0 = im.view(-1, 512, 49)
        im_embed = torch.squeeze(torch.relu(self.drop(self.im_dim(im_embed_0))), 2) ###[batch_size, 512, 1]
        im_embed = torch.relu(self.drop(self.im_dim_1(im_embed))) ###[batch_size, 512, 1]

        return im_embed

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
