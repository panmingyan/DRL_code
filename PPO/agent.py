#!/usr/bin/env python
# coding=utf-8
'''
Author: P.M.Y
Date: 2022-5-25
'''
import os
import numpy as np
import torch 
import torch.optim as optim
from torch.distributions import Normal
from model import Actor,Critic
from model import Actor2,Critic2
from memory import PPOMemory

class PPO:
    def __init__(self, state_dim, action_dim,cfg):
        self.gamma = cfg.gamma
        self.continuous = cfg.continuous 
        self.policy_clip = cfg.policy_clip
        self.n_epochs = cfg.n_epochs
        self.gae_lambda = cfg.gae_lambda
        self.device = cfg.device
        self.actor = Actor(state_dim, action_dim,cfg.hidden_dim).to(self.device)
        self.critic = Critic(state_dim,cfg.hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PPOMemory(cfg.batch_size)
        self.loss = 0

    def choose_action(self, state):
        state = torch.tensor(np.array([state])).to(self.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        if self.continuous:
            action = torch.tanh(action)
        else:
            action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return action, probs, value

    def update(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,reward_arr, dones_arr, batches = self.memory.sample()
            values = vals_arr[:]
            ### compute advantage ###
            # 计算优势函数
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)
            ### SGD ###
            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                if self.continuous:
                    actions = torch.tensor(action_arr[batch]).to(self.device)
                else:
                    actions = torch.tensor(action_arr[batch]).to(self.device)
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5*critic_loss
                self.loss  = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()

    def test_update(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,reward_arr, dones_arr, batches = self.memory.sample()
            values = vals_arr[:]
            ### compute advantage ###
            # 计算优势函数
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)
            ### SGD ###
            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                if self.continuous:
                    actions = torch.tensor(action_arr[batch]).to(self.device)
                else:
                    actions = torch.tensor(action_arr[batch]).to(self.device)
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                print(actions)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5*critic_loss
                self.loss  = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()

    def save(self,path):
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint= os.path.join(path, 'ppo_critic.pt')
        torch.save(self.actor.state_dict(), actor_checkpoint)
        torch.save(self.critic.state_dict(), critic_checkpoint)
    def load(self,path):
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint= os.path.join(path, 'ppo_critic.pt')
        self.actor.load_state_dict(torch.load(actor_checkpoint)) 
        self.critic.load_state_dict(torch.load(critic_checkpoint))  


class PPO2:
    def __init__(self, state_dim, action_dim,cfg):
        self.gamma = cfg.gamma
        self.continuous = cfg.continuous
        self.policy_clip = cfg.policy_clip
        self.n_epochs = cfg.n_epochs
        self.gae_lambda = cfg.gae_lambda
        self.device = cfg.device
        self.actor = Actor2(state_dim, action_dim,cfg.hidden_dim).to(self.device)
        self.critic = Critic2(state_dim,cfg.hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PPOMemory(cfg.batch_size)
        self.loss = 0

    def choose_action(self, state):
        state = torch.tensor(np.array([state])).to(self.device)
        (mu, sigma) = self.actor(state)
        dist = Normal(mu, sigma)
        value = self.critic(state)
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        if self.continuous:
            action = torch.tanh(action)
        else:
            action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return action, probs, value

    def update(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,reward_arr, dones_arr, batches = self.memory.sample()
            values = vals_arr[:]
            ### compute advantage ###
            # 计算优势函数
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)
            ### SGD ###
            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = torch.tensor(np.array(state_arr[batch])).to(self.device)
                old_probs = torch.tensor(np.array(old_prob_arr[batch])).to(self.device)
                actions = action_arr[batch]
                actions = [val.item() for val in actions]
                if self.continuous:
                    actions = torch.tensor(actions).to(self.device)
                else:
                    actions = torch.tensor(action_arr[batch]).to(self.device)
                (mu, sigma) = self.actor(states)
                dist = Normal(mu, sigma)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5*critic_loss
                self.loss  = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()
    def save(self,path):
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint= os.path.join(path, 'ppo_critic.pt')
        torch.save(self.actor.state_dict(), actor_checkpoint)
        torch.save(self.critic.state_dict(), critic_checkpoint)
    def load(self,path):
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint= os.path.join(path, 'ppo_critic.pt')
        self.actor.load_state_dict(torch.load(actor_checkpoint))
        self.critic.load_state_dict(torch.load(critic_checkpoint))

