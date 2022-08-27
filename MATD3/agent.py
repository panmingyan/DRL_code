#!/usr/bin/env python
# coding=utf-8

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import copy

from utils import write_txt_file
from replay_buffer import ReplayBuffer


# Ornstein–Uhlenbeck noise for exploration
# class OUActionNoise():
#     def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):    # mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None
#         self.theta = theta
#         self.mu = mu
#         self.sigma = sigma
#         self.dt = dt
#         self.x0 = x0
#         self.reset()
#
#     def __call__(self):
#         x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
#             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
#         self.x_prev = x
#
#         return x
#
#     def reset(self):
#         self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class Actor(nn.Module):
    def __init__(self, ob_dim, action_dim, hidden1_dim, hidden2_dim, init_w=3e-3):
        super(Actor, self).__init__()  
        self.linear1 = nn.Linear(ob_dim, hidden1_dim)
        self.linear2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.linear3 = nn.Linear(hidden2_dim, action_dim)
        # 随机初始化为较小的值
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))                         # 使动作值输出范围限定在[-1, 1]
        # x = torch.sigmoid(self.linear3(x))                    # 使动作值输出范围限定在[0, 1]
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

        self.linear4 = nn.Linear(state_dim + action_dim, hidden1_dim)  # 输入维度为 state_dim + action_dim
        self.linear5 = nn.Linear(hidden1_dim, hidden2_dim)
        self.linear6 = nn.Linear(hidden2_dim, 1)
        # 随机初始化为较小的值
        self.linear6.weight.data.uniform_(-init_w, init_w)
        self.linear6.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        # 按维数1拼接
        state = torch.cat(state, dim=1)
        action = torch.cat(action, dim=1)

        x = torch.cat((state, action), dim=1)
        q1 = F.relu(self.linear1(x))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        q2 = F.relu(self.linear4(x))
        q2 = F.relu(self.linear5(q2))
        q2 = self.linear6(q2)

        return q1, q2

    def Q1(self, state, action):
        # 按维数1拼接
        state = torch.cat(state, dim=1)
        action = torch.cat(action, dim=1)

        x = torch.cat((state, action), dim=1)

        q1 = F.relu(self.linear1(x))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)
        return q1


class MATD3:
    def __init__(self, ob_dim_n, action_dim_n, state_dim, cfg, agent_id):
        self.device = cfg.device
        self.agent_id = agent_id
        self.ob_dim = ob_dim_n[agent_id]
        self.action_dim = action_dim_n[agent_id]
        self.critic = Critic(state_dim, sum(action_dim_n), cfg.hidden1_dim, cfg.hidden2_dim).to(self.device)
        self.actor = Actor(self.ob_dim, self.action_dim, cfg.hidden1_dim, cfg.hidden2_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_actor = copy.deepcopy(self.actor)

        # 复制参数到目标网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBuffer(cfg, self.ob_dim, self.action_dim)        # 配置经验回放
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau                         # 软更新参数
        self.gamma = cfg.gamma
        self.use_grad_clip = True                            # 是否使用梯度剪枝
        self.actions_count = 0
        self.total_it = 0
        self.policy_freq = cfg.policy_freq                   # 策略网络更新频率
        self.epsilon_start = cfg.epsilon_start               # e-greedy策略中初始epsilon
        self.epsilon_end = cfg.epsilon_end                   # e-greedy策略中的终止epsilon
        self.epsilon_decay = cfg.epsilon_decay               # e-greedy策略中epsilon的衰减率
        # self.noise = OUActionNoise(mu=np.zeros(self.action_dim))  # 动作噪音
        self.policy_noise = cfg.policy_noise  # 目标动作噪音
        self.noise_clip = cfg.noise_clip

    def choose_action(self, ob, env, agent_name, mode):
        self.actions_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.actions_count / self.epsilon_decay)

        ob = torch.FloatTensor(ob.reshape(1, -1)).to(self.device)
        if mode == 'train':
            if np.random.uniform() > self.epsilon:
                action = self.actor(ob)
            else:
                action = torch.Tensor(env.action_space[agent_name].sample()).to(self.device)
            # action = action + torch.tensor(self.noise(), dtype=torch.float).to(self.device)     # 增加动作噪音
        elif mode == 'test':
            action = self.actor(ob)
        return action.detach().cpu().numpy().flatten().clip(-1, 1)

    def update(self, replay_buffer, agent_n):
        if replay_buffer.current_size < self.batch_size:                   # 当 memory 中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        # batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample(self.device)

        batch_obs_n = []
        batch_a_n = []
        batch_r_n = []
        batch_obs_next_n = []
        batch_done_n = []

        batch_index = replay_buffer.make_index()
        for agent_id in range(len(agent_n)):
            batch_obs, batch_a, batch_r, batch_obs_next, batch_done = agent_n[agent_id].memory.sample(batch_index, self.device)

            batch_obs_n.append(batch_obs)
            batch_a_n.append(batch_a)
            batch_r_n.append(batch_r)
            batch_obs_next_n.append(batch_obs_next)
            batch_done_n.append(batch_done)

        # 计算 target_Q
        with torch.no_grad():  # target_Q 无梯度
            # 根据target_actor网络确定next actions
            # 目标策略平滑
            noise = (torch.randn_like(batch_a_n[0]) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            batch_a_next_n = [(agent.target_actor(batch_obs_next) + noise).clip(-1, 1) for agent, batch_obs_next in zip(agent_n, batch_obs_next_n)]
            Q1_next, Q2_next = self.target_critic(batch_obs_next_n, batch_a_next_n)
            Q_next = torch.min(Q1_next, Q2_next)
            target_Q = batch_r_n[self.agent_id] + self.gamma * (1 - batch_done_n[self.agent_id]) * Q_next  # shape:(batch_size,1)

        # 计算 current_Q
        current_Q1, current_Q2 = self.critic(batch_obs_n, batch_a_n)                        # shape:(batch_size,1)
        critic_loss = F.mse_loss(target_Q, current_Q1) + F.mse_loss(target_Q, current_Q2)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # 延迟策略更新
        self.total_it += 1
        if self.total_it % self.policy_freq == 0:

            # Reselect the actions of the agent corresponding to 'agent_id'，the actions of other agents remain unchanged
            batch_a_n[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
            actor_loss = -self.critic.Q1(batch_obs_n, batch_a_n).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.soft_tau * param.data + (1 - self.soft_tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.soft_tau * param.data + (1 - self.soft_tau) * target_param.data)

    def save(self, path, ep_count, max_train_ep, max_ave_rew, agent_id):
        model_actor = "Actor_agent{}.pt".format(agent_id)
        model_critic = "Critic_agent{}.pt".format(agent_id)
        torch.save(self.actor.state_dict(), path + '/model/' + model_actor)
        torch.save(self.critic.state_dict(), path + '/model/' + model_critic)

        record = str(ep_count) + '/' + str(max_train_ep) + '\n' + '平均奖励值为：' + str(max_ave_rew)
        write_txt_file(path + '/model/' + 'log.txt', record)      # 记录保存模型时的回合数

    def load(self, path, agent_id):
        model_actor = "Actor_agent{}.pt".format(agent_id)
        model_critic = "Critic_agent{}.pt".format(agent_id)
        self.actor.load_state_dict(torch.load(path + '/model/' + model_actor))
        self.target_actor.load_state_dict(torch.load(path + '/model/' + model_actor))
        self.critic.load_state_dict(torch.load(path + '/model/' + model_critic))
        self.target_critic.load_state_dict(torch.load(path + '/model/' + model_critic))


