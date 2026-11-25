from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models as tmodels
import torch
class VGG11Slim(nn.Module): 
    """Extends VGG11 with a fewer layers in the classifier.
    
    Slimmer version of vgg11 model with fewer layers in classifier.
    """
    
    def __init__(self, hiddim, dropout=True, dropoutp=0.2, pretrained=True, freeze_features=True):
        """Initialize VGG11Slim Object.

        Args:
            hiddim (int): Hidden dimension size
            dropout (bool, optional): Whether to apply dropout to output of ReLU. Defaults to True.
            dropoutp (float, optional): Dropout probability. Defaults to 0.2.
            pretrained (bool, optional): Whether to instantiate VGG11 from Pretrained. Defaults to True.
            freeze_features (bool, optional): Whether to keep VGG11 features frozen. Defaults to True.
        """
        super(VGG11Slim, self).__init__()
        self.hiddim = hiddim
        self.nn = nn.Linear(512 * 7 * 7, hiddim)
        #self.layer = nn.Linear(512, hiddim)
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        features = self.nn(x)
        #print("features ",features.size())
        return features

