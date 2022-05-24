#!/usr/bin/env python
# coding=utf-8
'''
Author: P.M.Y
Date: 2022-5-22
'''
import sys, os

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加父路径到系统路径sys.path

import gym
import numpy as np
import torch
import datetime
from itertools import count

from agent import Policy
from common.utils import plot_rewards
from common.utils import save_results, make_dir

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class PGConfig:
    def __init__(self):
        self.algo_name = "PolicyGradient"  # 算法名称
        self.env_name = 'MountainCar-v0'  # 环境名称
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.train_eps = 1000  # 训练的回合数
        self.test_eps = 10  # 测试的回合数
        self.batch_size = 8
        self.lr = 0.02  # 学习率
        self.gamma = 0.995
        self.hidden_dim = 20  # dimmension of hidden layer
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # check gpu


class PlotConfig:
    ''' 绘图相关参数设置
    '''

    def __init__(self) -> None:
        self.algo_name = cfg.algo_name  # 算法名称
        self.env_name = cfg.env_name  # 环境名称
        self.device = cfg.device  # 检测GPU
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片


def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env_name)
    env = env.unwrapped
    # env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = Policy(state_dim, action_dim, cfg)
    return env, agent


def train(cfg, env, agent):
    print('Start to train !')
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    state_pool = []  # 存放每batch_size个episode的state序列
    action_pool = []
    reward_pool = []
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        for _ in count():
            action = agent.choose_action(state)  # 根据当前环境state选择action
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                reward = 0
            state_pool.append(state)
            action_pool.append(float(action))
            reward_pool.append(reward)
            state = next_state
            if done:
                print('Episode:', (i_ep + 1), ' Reward:', ep_reward)
                break
        if (i_ep + 1) % cfg.batch_size == 0:  # 每个batch更新一次参数
            agent.update(reward_pool, state_pool, action_pool)
            state_pool = []  # 每个episode的state
            action_pool = []
            reward_pool = []
        rewards.append(ep_reward)
        if (i_ep + 1) % 100 == 0:
            print("Episode:{}/{}: Ave_Reward:{}".format(i_ep + 1, cfg.train_eps, np.mean(rewards)) + '\n')
        if ma_rewards:
            ma_rewards.append(
                0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('complete training！')
    return rewards, ma_rewards


def eval(cfg, env, agent):
    print('Start to eval !')
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        env.render()
        ep_reward = 0
        for _ in count():
            action = agent.choose_action(state)  # 根据当前环境state选择action
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            env.render()
            if done:
                reward = 0
            state = next_state
            if done:
                print('Episode:', (i_ep + 1), ' Reward:', ep_reward)
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('complete evaling！')
    return rewards, ma_rewards


if __name__ == "__main__":
    cfg = PGConfig()
    plot_cfg = PlotConfig()

    # train
    env, agent = env_agent_config(cfg, seed=1)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")  # 画出结果

    # eval
    env, agent = env_agent_config(cfg, seed=10)
    agent.load(path=cfg.model_path)
    rewards, ma_rewards = eval(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='eval', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="eval")  # 画出结果