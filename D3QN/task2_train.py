#!/usr/bin/env python
# coding=utf-8
'''
Author: P.M.Y
Date: 2022-05-31
'''

import sys, os

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import gym
import numpy as np
import torch
import datetime

from common.utils import save_results, make_dir
from common.utils import plot_rewards
from d3qn import D3QN

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
train_num = 'train_1'
max_ave_reward_ep = 0


class Config:
    def __init__(self):
        ################################## 环境超参数 ###################################
        self.algo_name = 'D3QN'  # 算法名称
        self.env_name = 'LunarLander-v2'  # 环境名称
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = 700  # 训练的回合数
        ################################################################################

        ################################## 算法超参数 ###################################
        self.gamma = 0.99  # 强化学习中的折扣因子
        self.epsilon_start = 1  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 1000  # e-greedy策略中epsilon的衰减率
        self.lr = 0.00075  # 学习率
        self.memory_capacity = 100000  # 经验回放的容量
        self.batch_size = 64  # mini-batch SGD中的批量大小
        self.target_update = 4  # 目标网络的更新频率
        self.hidden_dim = 256  # 网络隐藏层
        ################################################################################

        ################################# 保存结果相关参数 ##############################
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + train_num + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + train_num + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片
        ################################################################################


def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env_name)
    # env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = D3QN(state_dim, action_dim, cfg)
    return env, agent


def train(cfg, env, agent):
    print('开始训练!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    ave_reward = 0  # 记录平均奖励
    max_ave_reward = -9999 # 记录最大平均奖励
    max_last_10_reward = -9999 # 记录最大后10平均奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            agent.update()
            if done:
                break
        ep_reward = round(ep_reward, 1)  # 保留小数点后一位
        rewards.append(ep_reward)
        print('Episode:', (i_ep + 1), ' Reward:', ep_reward)
        ave_reward = round(np.mean(rewards), 3)
        if ma_rewards:
            ma_rewards.append(
                0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) >= 20:
            last_10_reward = round(np.mean(rewards[-10:]), 3)
            if last_10_reward >= max_last_10_reward:
                max_10_reward_ep = i_ep + 1
                torch.save(agent.target_net.state_dict(), cfg.model_path + 'my10model_{}.pth'.format(max_10_reward_ep))
                print('The maximum 10reward is {}.'.format(last_10_reward))
                print('my10model_{} is saved!'.format(max_10_reward_ep) + '\n')
                max_last_10_reward = last_10_reward
            else:
                print('The maximum 10reward is {}.'.format(max_last_10_reward))
                print('The episode of the maximum 10reward is {}.'.format(max_10_reward_ep) + '\n')
        if (i_ep + 1) % 50 == 0:
            print('Episode:{}/{}, Ave_Reward:{}'.format(i_ep + 1, cfg.train_eps, ave_reward) + '\n')
            # 画出结果
            save_results(rewards, ma_rewards, tag='train',
                         path=cfg.result_path)  # 保存结果
            plot_rewards(rewards, ma_rewards, cfg, tag="train")  # 画出结果
            # 保存最优模型
            if ave_reward >= max_ave_reward:
                max_ave_reward_ep = i_ep + 1
                torch.save(agent.target_net.state_dict(), cfg.model_path + 'mymodel_{}.pth'.format(max_ave_reward_ep))
                print('The maximum reward is {}.'.format(ave_reward))
                print('model_{} is saved!'.format(max_ave_reward_ep) + '\n')
                max_ave_reward = ave_reward
            else:
                print('The maximum reward is {}.'.format(max_ave_reward))
                print('The episode of the maximum reward is {}.'.format(max_ave_reward_ep) + '\n')
        if i_ep % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
    print('完成训练！')
    env.close()
    return





if __name__ == "__main__":
    cfg = Config()
    # 训练
    make_dir(cfg.result_path, cfg.model_path)  # 创建保存结果和模型路径的文件夹

    env, agent = env_agent_config(cfg)
    train(cfg, env, agent)
