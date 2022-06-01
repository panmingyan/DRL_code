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
from common.utils import plot_rewards, plot_rewards2
from d3qn import D3QN
from task2_train import Config


def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env_name)
    # env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = D3QN(state_dim, action_dim, cfg)
    return env, agent



def test(cfg, env, agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    ############# 由于测试不需要使用epsilon-greedy策略，所以相应的值设置为0 ###############
    cfg.epsilon_start = 0.0  # e-greedy策略中初始epsilon
    cfg.epsilon_end = 0.0  # e-greedy策略中的终止epsilon
    ################################################################################
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励

    for i_ep in range(test_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
            if done:
                break
        ep_reward = round(ep_reward, 1)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print('Episode:{}/{}   reward:{}'.format(i_ep + 1, test_eps, ep_reward))
    if (i_ep + 1) == test_eps:
        print('\nAve_Reward:{}'.format(np.mean(rewards)) + '\n')
    print('完成测试！')
    env.close()
    return rewards, ma_rewards


if __name__ == "__main__":
    test_eps = 20   # 测试的回合数

    # 测试
    cfg = Config()
    env, agent = env_agent_config(cfg)
    result_path = curr_path + "/outputs/" + cfg.env_name + \
                  '/' + 'train_1' + '/results/'
    agent.target_net.load_state_dict(torch.load('./outputs/LunarLander-v2/train_1/models/my10model_393.pth'))
    for target_param, param in zip(agent.target_net.parameters(), agent.policy_net.parameters()):
        param.data.copy_(target_param.data)
    rewards, ma_rewards = test(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='test',
                 path=result_path)  # 保存结果
    plot_rewards2(rewards, ma_rewards, cfg, tag="test")  # 画出结果


