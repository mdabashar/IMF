from os import path
from collections import OrderedDict
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import defaultdict

class MultiModalGBClassifier(nn.Module):

    def __init__(
        self,
        modalities,
        num_class = 101,
        hidden_dim=512,
        embed_dim=256,
        encoder_types=['mlp','mlp'],
        input_dims=768,
        device="mps", dropout=0.0, depth=1, activation="relu", normalization="none"
    ):
        super().__init__()

        self.modalities = modalities
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_class = num_class
        self.device = device
        self.dropout, self.depth = dropout, depth
        self.activation, self.normalization = activation, normalization

        # -------------------------
        # Encoders
        # -------------------------
        self.encoders = nn.ModuleDict()
        for m in modalities:
            self.encoders[m] = self.build_encoder(
                encoder_types[m],
                input_dims[m]
            )

        # -------------------------
        # Embeddings
        # -------------------------
        self.embeddings = nn.ModuleDict({
            m: self.build_block(hidden_dim, embed_dim, 1)
            for m in modalities
        })

        # -------------------------
        # Classifier
        # -------------------------
        self.fc_out = nn.Linear(embed_dim, num_class)

        # -------------------------
        # Dynamic layers
        # -------------------------
        self.additional_layers = nn.ModuleDict({
            m: nn.ModuleList() for m in modalities
        })

        self.relu = nn.ReLU()

        self.to(device)
        
    def build_encoder(self, encoder_type, input_dim):

        if encoder_type == "mlp":
            return self.build_block(input_dim, self.hidden_dim, self.depth)

        else:
            raise ValueError(f"Unknown encoder {encoder_type}")

    def build_block(self, input_dim, output_dim, depth):
        layers=[]; current=input_dim
        act = nn.GELU if self.activation == "gelu" else nn.ReLU
        for _ in range(depth):
            layers.append(nn.Linear(current, output_dim))
            if self.normalization == "layernorm": layers.append(nn.LayerNorm(output_dim))
            layers.extend((act(), nn.Dropout(self.dropout)))
            current=output_dim
        return nn.Sequential(*layers)
            
    def forward(self, inputs):
        """
        inputs = {
            "m1": tensor,
            "m2": tensor,
            ...
        }
        """

        features = {}

        for m in self.modalities:
            features[m] = self.encoders[m](inputs[m])

        return features 
    
    def classifier(self, x, modality):

        base = self.embeddings[modality](x)
        feature = self.fc_out(base)

        o_fea = feature
        add_fea = None

        for i, layer in enumerate(self.additional_layers[modality]):

            add = self.relu(layer(x))
            add_fea = self.fc_out(add)

            feature = feature + add_fea

            if i < len(self.additional_layers[modality]) - 1:
                o_fea = feature
        return feature, o_fea, add_fea
        
    def add_layer(self, modality):

        new_layer = nn.Linear(self.hidden_dim, self.embed_dim).to(self.device)

        nn.init.xavier_normal_(new_layer.weight)
        nn.init.constant_(new_layer.bias, 0)

        self.additional_layers[modality].append(new_layer)
