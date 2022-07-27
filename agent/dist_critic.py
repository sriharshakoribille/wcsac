# Distributional Critics
import torch
import torch.nn as nn
import numpy as np

class QR_SafetyCritic(nn.Module):
    """Quantile Regression Safety Critic
        Args:
            obs_dim: state dimension
            action_dim: action dimensions (countinous action space)
            hidden_dim: -
            num_qunatiles: number of quantiles to approximate quantile distribution (32 in paper)
            risk_level: Risk Level
    """
    def __init__(self, obs_dim, action_dim, hidden_dim, num_quantiles=32):
        super(QR_SafetyCritic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_q = num_quantiles
        
        self.head = nn.Linear(self.obs_dim + self.action_dim, hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, self.num_q)
        weight_init([self.head, self.lin1])
    
    def forward(self, obs, action):
        """Forward fct of QR-safety critic
        Args:
            input: (state, action)
        Returns:
            Quantile distribution of cost / approximated cvar
        """
        obs_action = torch.cat([obs, action], dim=-1)
        x = torch.relu(self.head(obs_action))
        x = torch.relu(self.lin1(x))
        out = self.lin2(x)
        return out.view(obs_action.shape[0], self.num_q, 1)


class IQN_SafetyCritic(nn.Module):
    """Implicit Quantile Network Safety Critic"""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(IQN_SafetyCritic, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.K = 32  # number of samples for policy -> not used, since we have actor
        self.N = 8  # number of quantile samples
        self.n_cos = 64
        self.hidden_dim = hidden_dim
        
        # Start from 0 (according to paper)
        self.pis = torch.FloatTensor([np.pi*i for i in range(self.n_cos)]).view(1,1,self.n_cos) 

        self.head = nn.Linear(obs_dim+action_dim, hidden_dim) 
        self.cos_embedding = nn.Linear(self.n_cos, hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)
        # weight_init([self.head, self.lin1])

    def calc_cos(self, batch_size, n_tau=8, risk_level=1):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1) * risk_level  + (1-risk_level)  # (batch_size, n_tau, 1)

        cos = torch.cos(taus*self.pis)
        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos, taus
    
    def forward(self, obs, action, num_tau=8, risk_level=1):
        """
        Quantile Calculation depending on the number of tau
        """
        batch_size = obs.shape[0]
        obs_action = torch.cat([obs, action], dim=-1)
        
        # cosine embedding
        cos, taus = self.calc_cos(batch_size, num_tau, risk_level=risk_level)  # cos (bs, n_tau, n_cos), tau (bs, n_tau, 1)
        cos = cos.view(batch_size*num_tau, self.n_cos) # cos (bs*n_tau, n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.hidden_dim) # (bs, n_tau, hidden_dim)
        # state-action embedding
        x = torch.relu(self.head(obs_action))  # (bs, hidden_dim)
        # combining embdeddings
        # x has shape (batch, hidden_dim) for multiplication –> reshape to (batch, 1, hidden_dim)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.hidden_dim)
        x = torch.relu(self.lin1(x))
        out = self.lin2(x)
        
        return out.view(batch_size, num_tau, 1), taus


def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')