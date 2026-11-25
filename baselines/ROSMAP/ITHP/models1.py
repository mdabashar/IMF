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
             
            nn.Linear(200, 2*length),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(2*self.length, 2*self.length * 2),
        )
        self.MLP1 = nn.Sequential(
            nn.Linear(2*self.length, 2*length),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(2*self.length, 200),
            nn.Sigmoid(),
            nn.Dropout(self.drop_prob),
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(2*self.length, self.length),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.length, self.length * 2),
        )

        self.MLP2 = nn.Sequential(
            nn.Linear(self.length, self.length),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.length, 200),
            nn.Sigmoid(),
            nn.Dropout(self.drop_prob),
        )

        self.criterion = nn.MSELoss()
        self.early = nn.Linear(length, 2)


    def forward(self, m1, m2, m3):
        
        m1 = (m1 - m1.min(dim=0, keepdim=True)[0]) / (m1.max(dim=0, keepdim=True)[0] - m1.min(dim=0, keepdim=True)[0] + 1e-6)
        m2 = (m2 - m2.min(dim=0, keepdim=True)[0]) / (m2.max(dim=0, keepdim=True)[0] - m2.min(dim=0, keepdim=True)[0] + 1e-6)
        m3 = (m3 - m3.min(dim=0, keepdim=True)[0]) / (m3.max(dim=0, keepdim=True)[0] - m3.min(dim=0, keepdim=True)[0] + 1e-6)

        h1 = self.encoder1(m2)
        mu1, logvar1 = h1.chunk(2, dim=-1)
        kl_loss_0 = self.kl_loss(mu1, logvar1)
        b0 = self.reparameterise(mu1, logvar1)
        output1 = self.MLP1(b0)
        mse_0 = self.criterion(output1, m1)
        IB0 = kl_loss_0 + self.p_beta * mse_0

        b0_sample = b0
        h2 = self.encoder2(b0_sample)
        mu2, logvar2 = h2.chunk(2, dim=-1)
        kl_loss_1 = self.kl_loss(mu2, logvar2)
        b1 = self.reparameterise(mu2, logvar2)
        output2 = self.MLP2(b1)
        mse_1 = self.criterion(output2, m3)
        IB1 = kl_loss_1 + self.p_gamma * mse_1
        IB_total = IB0 + self.p_lambda * IB1

        clf = self.early(b1)

        return clf, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1


    def kl_loss(self, mu, logvar):
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_mean = torch.mean(kl_div)
        return kl_mean

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

