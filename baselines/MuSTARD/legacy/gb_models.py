import torch.nn as nn
import torch


class Unimodal(nn.Module):
    def __init__(self, length = 200, input_length=200, functional_p = 0.3):
        super().__init__()
        self.drop = nn.Dropout(p=functional_p)
        self.trans = nn.Linear(input_length, length)
        self.norm = nn.BatchNorm1d(length)
    def forward(self, md):
        md = torch.relu(self.trans(md))
        md = self.norm(md)
        return md

class MLP(torch.nn.Module):
    """Two layered perceptron."""

    def __init__(self, indim, outdim, dropout=False, dropoutp=0.1):

        super(MLP, self).__init__()
        self.fc = nn.Linear(indim, outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout

    def forward(self, x):

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
