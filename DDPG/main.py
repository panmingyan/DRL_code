#!/usr/bin/env python
# coding=utf-8

import sys, os
import gym
import datetime

from utils import *
from gym.wrappers import RescaleAction
from agent import DDPG


curr_path = os.path.dirname(os.path.abspath(__file__))        # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)                      # 父路径
sys.path.append(parent_path)                                  # 添加路径到系统路径sys.path


env_name = ['Pendulum-v1', 'LunarLanderContinuous-v2', 'MountainCarContinuous-v0'][0]       # 环境名称，gym新版本（约0.21.0之后）中Pendulum-v0改为Pendulum-v1
algo_name = 'DDPG'                              # 算法名称

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        # 检测GPU

start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 获取训练开始时间

outputs_path = curr_path + "/outputs/" + env_name + '/' + start_time          # 定义结果存储路径


# 配置超参数
class DDPGConfig:
    def __init__(self):
        self.algo_name = algo_name         # 算法名称
        self.env_name = env_name           # 环境名称
        self.device = device               # 训练所用设备
        self.train_eps = 1000               # 训练的回合数
        self.test_eps = 100                 # 测试的回合数
        self.gamma = 0.99                  # 折扣因子
        self.max_step = 100000             # 单回合最大步数
        self.critic_lr = 1e-3              # Critic网络的学习率
        self.actor_lr = 1e-4               # Actor网络的学习率
        self.memory_capacity = 1000000     # 经验回放的容量
        self.batch_size = 64               # mini-batch SGD中的批量大小
        # self.target_update = 2             # 目标网络的更新频率
        self.hidden1_dim = 256             # 网络隐藏层1维度
        self.hidden2_dim = 256             # 网络隐藏层2维度
        self.soft_tau = 1e-2               # 软更新参数
        self.ep_r = 15                    # 计算最后10轮平均奖励，确定是否保存模型
        self.seed = 666                    # 设置随机种子
        self.epsilon_start = 0.99             # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01               # e-greedy策略中的终止epsilon
        self.epsilon_decay = 60000         # e-greedy策略中epsilon的衰减率

# 训练
def train():
    print(f'环境：{env_name} | 算法：{algo_name} | 设备：{device}' + '\n')

    make_dir(outputs_path + '/img/', outputs_path + '/model/')  # 创建文件夹

    write_txt_file('outputs_path.txt', outputs_path)            # 将模型存放地址写入txt文件

    env = gym.make(env_name)                             # 创建环境
    env = RescaleAction(env, min_action=-1, max_action=1)

    state_dim = env.observation_space.shape[0]           # 状态维度
    action_dim = env.action_space.shape[0]               # 动作维度
    state_type = env.observation_space                   # 状态数据类型
    action_type = env.action_space                       # 动作数据类型
    state_high = env.observation_space.high  # 状态最大值
    state_low = env.observation_space.low                # 状态最小值
    print('Information'+'\n' + \
          'name: '+ env_name+'\n' + 'state_dim: '+ str(state_dim) +'\n' + 'action_dim: '+ str(action_dim) +'\n' \
          'state_type: '+ str(state_type) +'\n' + 'action_type: '+ str(action_type) +'\n' \
          'state_high: '+ str(state_high) +'\n' + 'state_low: '+ str(state_low))

    cfg = DDPGConfig()                                   # 模型参数

    agent = DDPG(state_dim, action_dim, cfg)             # 创建智能体

    total_step = 0                                       # 记录总步数
    epoch = []                                           # 记录轮数，方便画图
    rewards = []                                         # 记录所有回合的奖励
    ave_rewards = []                                     # 记录平均奖励值
    ma_rewards = []                                      # 记录所有回合的 滑动平均奖励
    max_ave_rew = -float('inf')                          # 后几轮最大平均奖励值

    set_seed(env, cfg.seed)                              # 设置随机种子

    # 统计一下运行时间
    starttime = datetime.datetime.now()
    print('\n训练开始时间：', starttime)

    print('开始训练！')
    for ep in range(cfg.train_eps):
        ep_reward = 0

        state = env.reset()                              # 重置环境，获得初始状态
        for step in range(cfg.max_step):
            total_step += 1
            action = agent.choose_action(state, env, mode='train')

            next_state, reward, done, _ = env.step(action)     # (observation, reward, terminated, truncated, info)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()                                                   # 单步更新
            state = next_state

            if done:
                break

        epoch.append(ep + 1)                                                 # 记录轮数，方便画图
        ep_reward = round(ep_reward, 3)                                      # 单回合奖励保留后三位小数
        rewards.append(ep_reward)                                            # 记录本回合奖励
        if ma_rewards:                                                       # 记录本回合滑动平均奖励
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print("Episode:%d | Reward:%s | Ave_reward:%.3f | Steps:%d" \
            % (ep + 1, ep_reward, np.mean(rewards[-100:]), total_step))                # 打印结果

        if (ep + 1) % 20 == 0:                                               # 每20轮打印曲线图
            show(outputs_path, env_name, algo_name, epoch, rewards, ma_rewards, mode='train')

        if (ep + 1) >= cfg.ep_r:                                             # 后几轮平均奖励值较大时，保存模型
            if np.mean(rewards[-int(cfg.ep_r):]) > max_ave_rew:
                max_ave_rew = np.mean(rewards[-int(cfg.ep_r):])
                agent.save(outputs_path, ep + 1, cfg.train_eps, max_ave_rew)
                print('模型已保存！平均奖励值：%.3f' % max_ave_rew)

    print('训练完成！')
    endtime = datetime.datetime.now()

    all_time = endtime - starttime
    print("\n共计用时:", all_time)                                              # 统计训练用时

# 测试
def test():
    print(f'环境：{env_name} | 算法：{algo_name} | 设备：{device}' + '\n')

    env = gym.make(env_name, render_mode='human')        # 创建环境,并渲染环境
    env = RescaleAction(env, min_action=-1, max_action=1)

    state_dim = env.observation_space.shape[0]           # 状态维度
    action_dim = env.action_space.shape[0]               # 动作维度

    cfg = DDPGConfig()  # 设置参数

    agent = DDPG(state_dim, action_dim, cfg)             # 创建智能体

    with open('outputs_path.txt', encoding='utf-8') as f:     # 读取模型存放地址
        outputs_path = f.read()

    agent.load(outputs_path)      # 加载模型

    epoch = []  # 记录轮数，方便画图
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的 滑动平均奖励

    set_seed(env, cfg.seed)                              # 设置随机种子

    print('开始测试！\n')
    for ep in range(cfg.test_eps):
        ep_reward = 0

        state = env.reset()                              # 重置环境，获得初始状态

        for step in range(cfg.max_step):
            action = agent.choose_action(state, env, mode='test')

            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state

            if done:
                break

        epoch.append(ep + 1)  # 记录轮数，方便画图
        ep_reward = round(ep_reward, 3)  # 单回合奖励保留后三位小数
        rewards.append(ep_reward)  # 记录本回合奖励
        if ma_rewards:                                                       # 记录本回合滑动平均奖励
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print("Episode:%s | Reward:%s | Ave_reward:%.3f" % (ep + 1, ep_reward, np.mean(rewards)))  # 打印结果

    show(outputs_path, env_name, algo_name, epoch, rewards, ma_rewards, mode='test')
    print('平均测试奖励值为：', str(np.mean(rewards)))
    print('测试结束！')


if __name__ == "__main__":
    # 训练模型
    train()

    # 测试模型
    # test()


