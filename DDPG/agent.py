#!/usr/bin/env python
# coding=utf-8

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

from utils import write_txt_file


# Ornstein–Uhlenbeck noise for exploration
class OUActionNoise():
    def __init__(self, mu, sigma=0.07, theta=0.1, dt=1e-2, x0=None):    # mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # 经验回放的容量
        self.buffer = [] # 缓冲区
        self.position = 0 
    
    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # 随机采出小批量转移
        state, action, reward, next_state, done = zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state, done
    
    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1_dim, hidden2_dim, init_w=3e-3):
        super(Actor, self).__init__()  
        self.linear1 = nn.Linear(state_dim, hidden1_dim)
        self.linear2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.linear3 = nn.Linear(hidden2_dim, action_dim)
        # 随机初始化为较小的值
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))                     # 使动作值输出范围限定在[-1, 1]
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden1_dim, hidden2_dim, init_w=3e-3):
        super(Critic, self).__init__()
        
        self.linear1 = nn.Linear(state_dim + action_dim, hidden1_dim)   # 输入维度为 state_dim + action_dim
        self.linear2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.linear3 = nn.Linear(hidden2_dim, 1)
        # 随机初始化为较小的值
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        # 按维数1拼接
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DDPG:
    def __init__(self, state_dim, action_dim, cfg):
        self.device = cfg.device
        self.critic = Critic(state_dim, action_dim, cfg.hidden1_dim, cfg.hidden2_dim).to(cfg.device)
        self.actor = Actor(state_dim, action_dim, cfg.hidden1_dim, cfg.hidden2_dim).to(cfg.device)
        self.target_critic = Critic(state_dim, action_dim, cfg.hidden1_dim, cfg.hidden2_dim).to(cfg.device)
        self.target_actor = Actor(state_dim, action_dim, cfg.hidden1_dim, cfg.hidden2_dim).to(cfg.device)

        # 复制参数到目标网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau                         # 软更新参数
        self.gamma = cfg.gamma
        self.actions_count = 0
        self.epsilon_start = cfg.epsilon_start               # e-greedy策略中初始epsilon
        self.epsilon_end = cfg.epsilon_end                   # e-greedy策略中的终止epsilon
        self.epsilon_decay = cfg.epsilon_decay               # e-greedy策略中epsilon的衰减率
        self.noise = OUActionNoise(mu=np.zeros(action_dim))  # 动作噪音

    def choose_action(self, state, env, mode):
        self.actions_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.actions_count / self.epsilon_decay)

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        if mode == 'train':
            if np.random.uniform() > self.epsilon:
                action = self.actor(state)
            else:
                action = torch.Tensor(env.action_space.sample())
            action = action + torch.tensor(self.noise(), dtype=torch.float)     # 增加动作噪音
        elif mode == 'test':
            action = self.actor(state)
        return action.detach().cpu().numpy().flatten().clip(env.action_space.low, env.action_space.high)

    def update(self):
        if len(self.memory) < self.batch_size:                   # 当 memory 中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # 转变为张量
        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
       
        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()               # 更新网络

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()              # 更新网络

        # 软更新
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )

    def save(self, path, ep_count, max_train_ep, max_ave_rew):
        model_actor = "Actor.pt"
        model_critic = "Critic.pt"
        torch.save(self.actor.state_dict(), path + '/model/' + model_actor)
        torch.save(self.critic.state_dict(), path+ '/model/' + model_critic)

        record = str(ep_count) + '/' + str(max_train_ep) + '\n' + '平均奖励值为：' + str(max_ave_rew)
        write_txt_file(path + '/model/' + 'log.txt', record)      # 记录保存模型时的回合数

    def load(self, path):
        model_actor = "Actor.pt"
        model_critic = "Critic.pt"
        self.actor.load_state_dict(torch.load(path + '/model/' + model_actor))
        self.target_actor.load_state_dict(torch.load(path + '/model/' + model_actor))
        self.critic.load_state_dict(torch.load(path + '/model/' + model_critic))
        self.target_critic.load_state_dict(torch.load(path + '/model/' + model_critic))
