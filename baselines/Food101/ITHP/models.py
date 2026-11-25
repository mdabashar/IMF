import torch.nn as nn
import torch
import torch.nn.functional as F


class IB(nn.Module):
    def __init__(self, dim=200, length = 200, functional_p = 0.3, p_gamma = 8, p_beta = 32, p_lambda = 0.3):
        super().__init__()

        self.drop_prob = functional_p
        self.length = length
        self.p_gamma = p_gamma 
        self.p_beta = p_beta 
        self.p_lambda = p_lambda
        self.encoder1 = nn.Sequential(
             
            nn.Linear(768, length),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.length, self.length * 2),
        )
        self.MLP1 = nn.Sequential(
            #nn.Linear(self.length, length),
            #nn.ReLU(),
            nn.Linear(self.length, length),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.length, 768),
            nn.Sigmoid(),
            nn.Dropout(self.drop_prob),
        )
        
        self.criterion = nn.MSELoss()
        self.early = nn.Linear(length, 101)


    def forward(self, m1, m2):
        
        m1 = (m1 - m1.min(dim=0, keepdim=True)[0]) / (m1.max(dim=0, keepdim=True)[0] - m1.min(dim=0, keepdim=True)[0] + 1e-6)
        m2 = (m2 - m2.min(dim=0, keepdim=True)[0]) / (m2.max(dim=0, keepdim=True)[0] - m2.min(dim=0, keepdim=True)[0] + 1e-6)


        h1 = self.encoder1(m2)
        mu1, logvar1 = h1.chunk(2, dim=-1)
        kl_loss_0 = self.kl_loss(mu1, logvar1)
        b0 = self.reparameterise(mu1, logvar1)
        output1 = self.MLP1(b0)
        mse_0 = self.criterion(output1, m1)
        IB0 = kl_loss_0 + self.p_beta * mse_0

        clf = self.early(b0)

        return clf, IB0, kl_loss_0, mse_0, 0, 0


    def kl_loss(self, mu, logvar):
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_mean = torch.mean(kl_div)
        return kl_mean

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

