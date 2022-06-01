#!/usr/bin/env python
# coding=utf-8
'''
Author: P.M.Y
Date: 2022-05-30
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import gym
import random
import math
import numpy as np

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
        state, action, reward, next_state, done =  zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state, done
    
    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)

class MLP(nn.Module):
    def __init__(self, state_dim,action_dim,hidden_dim=128):
        """ 初始化q网络，为全连接网络
            state_dim: 输入的特征数即环境的状态维度
            action_dim: 输出的动作维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim) # 输入层
        self.fc2 = nn.Linear(hidden_dim,hidden_dim) # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, action_dim) # 输出层
        
    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingNet, self).__init__()

        # 隐藏层
        self.hidden = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # 优势函数
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # 价值函数
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.hidden(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = Variable(torch.FloatTensor(state).unsqueeze(0))
                q_value = self.forward(state)
                action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)
        return action

class DoubleDQN:
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim  # 总的动作个数
        self.device = cfg.device  # 设备，cpu或gpu等
        self.gamma = cfg.gamma
        # e-greedy策略相关参数
        self.actions_count = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.policy_net = MLP(state_dim, action_dim,hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim,hidden_dim=cfg.hidden_dim).to(self.device)
        # target_net copy from policy_net
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        # self.target_net.eval()  # 不启用 BatchNormalization 和 Dropout
        # 可查parameters()与state_dict()的区别，前者require_grad=True
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.loss = 0
        self.memory = ReplayBuffer(cfg.memory_capacity)
        
    def choose_action(self, state):
        '''选择动作
        '''
        self.actions_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.actions_count / self.epsilon_decay)
        if random.random() > self.epsilon:
            with torch.no_grad():
                # 先转为张量便于丢给神经网络,state元素数据原本为float64
                # 注意state=torch.tensor(state).unsqueeze(0)跟state=torch.tensor([state])等价
                state = torch.tensor(
                    [state], device=self.device, dtype=torch.float32)
                # 如tensor([[-0.0798, -0.0079]], grad_fn=<AddmmBackward>)
                q_value = self.policy_net(state)
                # tensor.max(1)返回每行的最大值以及对应的下标，
                # 如torch.return_types.max(values=tensor([10.3587]),indices=tensor([0]))
                # 所以tensor.max(1)[1]返回最大值对应的下标，即action
                action = q_value.max(1)[1].item()  
        else:
            action = random.randrange(self.action_dim)
        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        # 从memory中随机采样transition
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        # convert to tensor
        state_batch = torch.tensor(
            state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(
            1)  # 例如tensor([[1],...,[0]]),unsqueeze(1)指升一维
        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float)  # tensor([1., 1.,...,1])
        next_state_batch = torch.tensor(
            next_state_batch, device=self.device, dtype=torch.float)
        
        done_batch = torch.tensor(np.float32(
            done_batch), device=self.device)  # 将bool转为float然后转为张量
        # 计算当前(s_t,a)对应的Q(s_t, a)
        q_values = self.policy_net(state_batch) 
        next_q_values = self.policy_net(next_state_batch)
        # 代入当前选择的action，得到Q(s_t|a=a_t)
        q_value = q_values.gather(dim=1, index=action_batch)
        '''以下是Nature DQN的q_target计算方式
        # 计算所有next states的Q'(s_{t+1})的最大值，Q'为目标网络的q函数
        next_q_state_value = self.target_net(
            next_state_batch).max(1)[0].detach()  # 比如tensor([ 0.0060, -0.0171,...,])
        # 计算 q_target
        # 对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        q_target = reward_batch + self.gamma * next_q_state_value * (1-done_batch[0])
        '''
        '''以下是Double DQN q_target计算方式，与NatureDQN稍有不同'''
        next_target_values = self.target_net(
            next_state_batch)
        # 选出Q(s_t‘, a)对应的action，代入到next_target_values获得target net对应的next_q_value，即Q’(s_t|a=argmax Q(s_t‘, a))
        next_target_q_value = next_target_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        q_target = reward_batch + self.gamma * next_target_q_value * (1-done_batch)
        self.loss = nn.MSELoss()(q_value, q_target.unsqueeze(1))  # 计算 均方误差loss
        # 优化模型
        self.optimizer.zero_grad()  # zero_grad清除上一步所有旧的gradients from the last step
        # loss.backward()使用backpropagation计算loss相对于所有parameters(需要gradients)的微分
        self.loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()  # 更新模型

    def save(self,path):
        torch.save(self.target_net.state_dict(), path+'checkpoint.pth')

    def load(self,path):
        self.target_net.load_state_dict(torch.load(path+'checkpoint.pth'))  
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


class D3QN:
    def __init__(self, state_dim, action_dim, cfg) -> None:
        self.action_dim = action_dim  # 总的动作个数
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.device = cfg.device
        self.loss_history = []  # 记录loss的变化
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
                                         (cfg.epsilon_start - cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.policy_net = DuelingNet(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = DuelingNet(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):  # 复制参数到目标网络targe_net
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)  # 优化器
        self.memory = ReplayBuffer(cfg.memory_capacity)

    def choose_action(self, state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()  # 选择Q值最大的动作
        else:
            action = random.randrange(self.action_dim)
        return action


    def update(self):
        if len(self.memory) < self.batch_size:  # 当memory中不满足一个批量时，不更新策略
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = torch.tensor(state, device=self.device, dtype=torch.float)
        action = torch.tensor(action, device=self.device).unsqueeze(1)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float)
        next_state = torch.tensor(next_state, device=self.device, dtype=torch.float)
        done = torch.tensor(np.float32(done), device=self.device)
        q_values = self.policy_net(state)
        next_q_values = self.target_net(next_state)

        q_value = q_values.gather(dim=1, index=action)
        next_target_values = self.target_net(next_state)
        next_target_q_value = next_target_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        q_target = reward + self.gamma * next_target_q_value * (1 - done)
        self.loss = nn.MSELoss()(q_value, q_target.unsqueeze(1))  # 计算 均方误差loss
        # 优化模型
        self.optimizer.zero_grad()  # zero_grad清除上一步所有旧的gradients from the last step
        # loss.backward()使用backpropagation计算loss相对于所有parameters(需要gradients)的微分
        self.loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()  # 更新模型

        # next_q_value = next_q_values.max(1)[0]
        # expected_q_value = reward + gamma * next_q_value * (1 - done)
        #
        # loss = (q_value - expected_q_value.detach()).pow(2).mean()
        # self.loss_history.append(loss)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
