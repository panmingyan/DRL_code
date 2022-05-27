#!/usr/bin/env python
# coding=utf-8
'''
Author: P.M.Y
Date: 2022-5-25
'''
import numpy as np
import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class Actor(nn.Module):
    def __init__(self,state_dim, action_dim,
            hidden_dim):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
        )
    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

class Critic(nn.Module):
    def __init__(self, state_dim,hidden_dim):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
        )
    def forward(self, state):
        value = self.critic(state)
        return value


class Actor2(nn.Module):

    def __init__(self,state_dim, action_dim,
        hidden_dim):
        super(Actor2, self).__init__()
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.sigma_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = 2.0 * F.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma)

class Critic2(nn.Module):
    def __init__(self, state_dim,hidden_dim):
        super(Critic2, self).__init__()

        self.fc = nn.Linear(state_dim, 100)
        self.v_head = nn.Linear(100, 1)

    def forward(self, state):
        x = F.relu(self.fc(state))
        value = self.v_head(x)
        return value