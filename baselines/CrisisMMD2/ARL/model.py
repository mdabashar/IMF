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
        num_class=101,
        hidden_dim=512,
        embed_dim=256,
        encoder_types=None,
        input_dims=None,
        device="mps"
    ):
        super().__init__()

        self.modalities = modalities
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_class = num_class
        self.device = device

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
        # Embeddings（统一空间）
        # -------------------------
        self.embeddings = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(hidden_dim, embed_dim),
                nn.ReLU()
            )
            for m in modalities
        })

        # -------------------------
        # Unimodal heads
        # -------------------------
        self.heads = nn.ModuleDict({
            m: nn.Linear(embed_dim, num_class)
            for m in modalities
        })

        # -------------------------
        # Multimodal head（concat）
        # -------------------------
        self.head = nn.Linear(embed_dim * len(modalities), num_class)

        self.relu = nn.ReLU()

        self.to(device)

    def build_encoder(self, encoder_type, input_dim):

        if encoder_type == "mlp":
            return nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Unknown encoder {encoder_type}")

    def forward(self, inputs, arl_weights=None):
        """
        arl_weights: dict or None
        """
    
        features = {}
        embeds = {}
    
        # =========================
        # encode + embed
        # =========================
        for m in self.modalities:
            feat = self.encoders[m](inputs[m])
            emb = self.embeddings[m](feat)
    
            features[m] = feat
            embeds[m] = emb
    
        # =========================
        # ARL-weighted fusion（关键）
        # =========================
        if arl_weights is not None:
            concat = torch.cat([
                arl_weights[m] * embeds[m]
                for m in self.modalities
            ], dim=1)
        else:
            concat = torch.cat([
                embeds[m]
                for m in self.modalities
            ], dim=1)
    
        out_mm = self.head(concat)
    
        # =========================
        # unimodal outputs
        # =========================
        out_modalities = {
            m: self.heads[m](embeds[m])
            for m in self.modalities
        }
    
        return out_mm, out_modalities