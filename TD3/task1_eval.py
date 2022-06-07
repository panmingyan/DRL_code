#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-04-23 20:36:23
LastEditor: JiangJi
LastEditTime: 2021-04-28 10:14:33
Discription: 
Environment: 
'''
import sys,os
curr_path = os.path.dirname(__file__)
parent_path=os.path.dirname(curr_path) 
sys.path.append(parent_path) # add current terminal path to sys.path

import torch
import gym
import numpy as np
import datetime


from agent import TD3
from common.utils import plot_rewards
from common.utils import save_results,make_dir

# curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
curr_time = '20220607-162732'

class TD3Config:
	def __init__(self) -> None:
		self.algo_name = 'TD3'
		self.env_name = 'Pendulum-v1'
		self.seed = 0
		self.result_path = curr_path+"/results/" +self.env_name+'/'+curr_time+'/results/'  # path to save results
		self.model_path = curr_path+"/results/" +self.env_name+'/'+curr_time+'/models/'  # path to save models
		self.batch_size = 256 # Batch size for both actor and critic
		self.gamma = 0.99 # gamma factor
		self.lr = 0.0005 # Target network update rate 
		self.policy_noise = 0.2 # Noise added to target policy during critic update
		self.noise_clip = 0.5  # Range to clip target policy noise
		self.policy_freq = 2 # Frequency of delayed policy updates
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlotConfig(TD3Config):
	def __init__(self) -> None:
		super().__init__()
		self.result_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/results/'  # 保存结果的路径
		self.model_path = curr_path+"/outputs/" + self.env_name + \
            '/'+curr_time+'/models/'  # 保存模型的路径
		self.save = True # 是否保存图片
		
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval(env_name,agent, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)
	rewards,ma_rewards =[],[]
	for i_episode in range(eval_episodes):
		ep_reward = 0
		state, done = eval_env.reset(), False
		while not done:
			eval_env.render()
			action = agent.choose_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			ep_reward += reward
		print(f"Episode:{i_episode+1}, Reward:{ep_reward:.3f}")
		rewards.append(ep_reward)
		# 计算滑动窗口的reward
		if ma_rewards:
			ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
		else:
			ma_rewards.append(ep_reward) 
	return rewards,ma_rewards

if __name__ == "__main__":
	cfg  = TD3Config()
	plot_cfg = PlotConfig()
	env = gym.make(cfg.env_name)
	env.seed(cfg.seed) # Set seeds
	torch.manual_seed(cfg.seed)
	np.random.seed(cfg.seed)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])
	td3= TD3(state_dim,action_dim,max_action,cfg)
	cfg.model_path = './outputs/Pendulum-v1/20220607-162732/models/'
	cfg.result_path = './outputs/Pendulum-v1/20220607-162732/results/'
	td3.load(cfg.model_path)
	rewards,ma_rewards = eval(cfg.env_name,td3,cfg.seed)
	make_dir(cfg.result_path,cfg.model_path)
	save_results(rewards,ma_rewards,tag='eval',path=cfg.result_path)
	plot_rewards(rewards,ma_rewards, plot_cfg, tag="eval")