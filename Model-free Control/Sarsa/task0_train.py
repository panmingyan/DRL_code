#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-11 17:59:16
LastEditor: John
LastEditTime: 2021-05-06 17:12:37
Discription: 
Environment: 
'''
import sys, os
import time

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import datetime
import torch
import gym
import numpy as np
from envs.racetrack_env import RacetrackEnv
from agent import Sarsa
from common.utils import plot_rewards
from common.utils import save_results, make_dir

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time

class SarsaConfig:
    ''' parameters for Sarsa
    '''
    def __init__(self):
        self.algo = 'Sarsa'
        self.env = 'CliffWalking-v0' # 0 up, 1 right, 2 down, 3 left
        self.result_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/results/'  # path to save results
        self.model_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/models/'  # path to save models
        self.train_eps = 200
        self.test_eps = 10
        self.epsilon = 0.15 # epsilon: The probability to select a random action . 
        self.gamma = 0.9 # gamma: Gamma discount factor.
        self.lr = 0.2 # learning rate: step size parameter
        self.n_steps = 2000
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # check gpu

def env_agent_config(cfg,seed=1):
    # env = RacetrackEnv()
    env = gym.make('CliffWalking-v0')
    action_dim = 4
    agent = Sarsa(action_dim, cfg)
    return env, agent
        
def train(cfg,env,agent):
    rewards = []
    ma_rewards = []
    for i_episode in range(cfg.train_eps):
        # Print out which episode we're on, useful for debugging.
        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        state = env.reset()  # 得到一个初始状态
        ep_reward = 0
        while True:
        # for t in range(cfg.n_steps):
            action = agent.choose_action(state)
            next_state, reward, done, p = env.step(action)
            ep_reward+=reward
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action,done)
            state = next_state
            if done:
                break  
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        rewards.append(ep_reward)
        if (i_episode+1)%10==0:
            # print("Episode:{}/{}: Reward:{}".format(i_episode+1, cfg.train_eps,ep_reward))
            print("Episode:{}/{}: Ave_Reward:{}".format(i_episode + 1, cfg.train_eps, np.mean(rewards[-10:])) + '\n')
    return rewards,ma_rewards

def eval(cfg,env,agent):
    rewards = []
    ma_rewards = []
    print("------开始评估------\n")
    for i_episode in range(cfg.test_eps):
        # Print out which episode we're on, useful for debugging.
        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        print("第%d轮"%(i_episode +1) + '\n')
        state = env.reset()
        env.render()
        ep_reward = 0
        while True:
        # for t in range(cfg.n_steps):
            action = agent.choose_action(state)
            next_state, reward, done, p = env.step(action)
            ep_reward += reward
            state = next_state
            time.sleep(0.5)
            env.render()
            if done:
                break  
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        rewards.append(ep_reward)
        print("第{}轮回报：{}".format(i_episode+1, ep_reward) + '\n')
        if (i_episode+1)%10==0:
            print("Episode:{}/{}: Ave_Reward:{}".format(i_episode+1, cfg.test_eps, np.mean(rewards[-10:])) + '\n')
    print('Complete evaling！')
    return rewards,ma_rewards
        
if __name__ == "__main__":
    cfg = SarsaConfig()
    env,agent = env_agent_config(cfg,seed=1)
    rewards,ma_rewards = train(cfg,env,agent)
    make_dir(cfg.result_path,cfg.model_path)
    agent.save(path=cfg.model_path)            # 保存模型（Q表格）
    save_results(rewards,ma_rewards,tag='train',path=cfg.result_path)     # 保存奖励
    plot_rewards(rewards, ma_rewards, plot_cfg=cfg, tag="train")

    env,agent = env_agent_config(cfg,seed=10)
    agent.load(path=cfg.model_path)            # 载入模型（Q表格）
    rewards,ma_rewards = eval(cfg,env,agent)
    save_results(rewards,ma_rewards,tag='eval',path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, plot_cfg=cfg, tag="eval")
    
    

