#!/usr/bin/env python
# coding=utf-8
'''
Author: P.M.Y
Date: 2022-5-22
'''
import torch
from torch.distributions import Bernoulli
from torch.distributions import Categorical
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import adam
import numpy as np
from model import MLP

class PolicyGradient:                              # 用于task0
    
    def __init__(self, state_dim,cfg):
        self.gamma = cfg.gamma
        self.policy_net = MLP(state_dim,hidden_dim=cfg.hidden_dim)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=cfg.lr)
        self.batch_size = cfg.batch_size

    def choose_action(self,state):                 # 单输出
        
        state = torch.from_numpy(state).float()    # 转换成tensor
        state = Variable(state)                    # tensor变成variable才能进行反向传播
        probs = self.policy_net(state)
        m = Bernoulli(probs) # 伯努利分布
        action = m.sample()
        action = action.data.numpy().astype(int)[0] # 转为标量
        return action

    def test_choose_action(self, state):  # 单输出

        state = torch.from_numpy(state).float()  # 转换成tensor
        state = Variable(state)  # tensor变成variable才能进行反向传播
        probs = self.policy_net(state)
        m = Bernoulli(probs)  # 伯努利分布
        action = m.sample()
        print(action)
        action = action.data.numpy().astype(int)[0]  # 转为标量
        return action


    def update(self,reward_pool,state_pool,action_pool):                # 策略梯度核心算法
        # Discount reward
        running_add = 0
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add

        # Normalize reward
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

        # Gradient Desent
        self.optimizer.zero_grad()

        for i in range(len(reward_pool)):
            state = state_pool[i]
            action = Variable(torch.FloatTensor([action_pool[i]]))
            reward = reward_pool[i]
            state = Variable(torch.from_numpy(state).float())
            probs = self.policy_net(state)
            m = Bernoulli(probs)
            loss = -m.log_prob(action) * reward  # 加负号使得梯度上升
            # print(loss)
            loss.backward()                      # 反向传播更新参数
        self.optimizer.step()
    def save(self,path):
        torch.save(self.policy_net.state_dict(), path+'pg_checkpoint.pt')
    def load(self,path):
        self.policy_net.load_state_dict(torch.load(path+'pg_checkpoint.pt'))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

class Policy:                                # 用于task1

    def __init__(self, state_dim, action_dim, cfg):
        self.gamma = cfg.gamma
        self.hidden_dim = cfg.hidden_dim
        self.policy_net = nn.Sequential(
                            nn.Linear(state_dim, self.hidden_dim),
                            nn.ReLU(),
                            nn.Linear(self.hidden_dim, action_dim),
                            nn.Softmax())
        self.policy_net.apply(init_weights)
        self.optimizer = adam.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.batch_size = cfg.batch_size

    def choose_action(self, state):  # 单输出

        state = torch.from_numpy(state).float()  # 转换成tensor
        state = Variable(state)  # tensor变成variable才能进行反向传播
        probs = self.policy_net(state)
        m = Categorical(probs)
        action = m.sample()
        action = action.data.numpy().astype(int)  # 转为标量
        return action

    def update(self, reward_pool, state_pool, action_pool):  # 策略梯度核心算法
        # Discount reward
        running_add = 0
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add

        # Normalize reward
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

        # Gradient Desent
        self.optimizer.zero_grad()

        for i in range(len(reward_pool)):
            state = state_pool[i]
            action = Variable(torch.FloatTensor([action_pool[i]]))
            reward = reward_pool[i]
            state = Variable(torch.from_numpy(state).float())
            probs = self.policy_net(state)
            m = Categorical(probs)
            loss = -m.log_prob(action) * reward  # 加负号使得梯度上升
            # print(loss)
            loss.backward()  # 反向传播更新参数
        self.optimizer.step()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'pg_checkpoint.pt')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'pg_checkpoint.pt'))

