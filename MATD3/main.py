#!/usr/bin/env python
# coding=utf-8
# 查看cuda GPU占用率cmd命令：nvidia-smi -l 3

import sys, os
import gym
import pettingzoo
import datetime
import time

from utils import *
from Wrapper import mpe_wrapper_for_pettingzoo
from gym.wrappers import RescaleAction
from pettingzoo.mpe import simple_reference_v2, simple_spread_v2, simple_adversary_v2
from agent import MATD3


curr_path = os.path.dirname(os.path.abspath(__file__))        # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)                      # 父路径
sys.path.append(parent_path)                                  # 添加路径到系统路径sys.path


env_name = ['simple_reference_v2', 'simple_spread_v2', 'simple_adversary_v2'][0]         # 环境名称
algo_name = 'MATD3'                  # 算法名称

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        # 检测GPU

start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 获取训练开始时间

# outputs_path = curr_path + "/outputs/" + env_name + '/' + start_time          # 定义结果存储路径


# 配置超参数
class MATD3Config:
    def __init__(self):
        self.algo_name = algo_name         # 算法名称
        self.env_name = env_name           # 环境名称
        self.device = device               # 训练所用设备
        self.train_eps = 1000000           # 训练的回合数
        self.test_eps = 100                 # 测试的回合数
        self.gamma = 0.95                  # 折扣因子
        self.max_step = 100000             # 单回合最大步数
        self.critic_lr = 5e-4              # Critic网络的学习率
        self.actor_lr = 5e-4               # Actor网络的学习率
        self.memory_capacity = 100000      # (每个智能体)经验回放的容量
        self.batch_size = 1024             # mini-batch SGD中的批量大小
        self.policy_noise = 0.05           # 目标策略噪音
        self.noise_clip = 0.05             # 目标策略噪音范围
        self.policy_freq = 2               # 策略网络的更新频率
        self.hidden1_dim = 256             # 网络隐藏层1维度
        self.hidden2_dim = 128             # 网络隐藏层2维度
        self.soft_tau = 1e-2               # 软更新参数
        self.ep_r = 1000                     # 计算最后1000轮平均奖励，确定是否保存模型
        self.seed = 666                    # 设置随机种子
        self.epsilon_start = 0.99             # e-greedy策略中初始epsilon (接续训练时注意修改)
        self.epsilon_end = 0.01               # e-greedy策略中的终止epsilon
        self.epsilon_decay = 60000         # e-greedy策略中epsilon的衰减率

# 训练
def train():
    print(f'环境：{env_name} | 算法：{algo_name} | 设备：{device}' + '\n')

    outputs_path = curr_path + "/outputs/" + env_name + '/' + start_time  # 定义结果存储路径

    # 创建环境
    # env = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=True)
    # env = simple_adversary_v2.parallel_env(N=2, max_cycles=25, continuous_actions=True)
    env = simple_reference_v2.parallel_env(local_ratio=0.5, max_cycles=25, continuous_actions=True)
    env = mpe_wrapper_for_pettingzoo(env, continuous_actions=True)

    n = env.num_agents                                     # 环境中智能体的数量
    agents = env.agents                                    # 环境中智能体列表
    ob_dim_n = env.obs_shape_n                             # 每个智能体的观测维度
    action_dim_n = env.act_shape_n                         # 每个智能体的动作维度
    state_dim = env.state_space.shape[0]                   # 全部智能体的状态维度
    ob_type = env.observation_space                        # 每个智能体的观测数据类型
    action_type = env.action_space                         # 每个智能体的动作数据类型
    ob_high = [env.observation_space[i].high for i in range(n)]         # 每个智能体观测最大值
    ob_low = [env.observation_space[i].low for i in range(n)]           # 每个智能体观测最小值
    action_high = [env.action_space[i].high for i in range(n)]          # 每个智能体动作最大值
    action_low = [env.action_space[i].low for i in range(n)]            # 每个智能体动作最大值
    print('Information'+'\n' + \
          'name: '+ env_name+'\n' + 'agent_num: ' + str(n) + '\n' + \
          'ob_dim: '+ str(ob_dim_n) +'\n' + 'action_dim: '+ str(action_dim_n) +'\n' + 'state_dim: ' + str(state_dim) +'\n' + \
          'ob_type: '+ str(ob_type) +'\n' + 'action_type: '+ str(action_type) + '\n' + \
          # 'ob_high: '+ str(ob_high) +'\n' + 'ob_low: '+ str(ob_low) + '\n' + \
          'action_high: '+ str(action_high) + '\n' + 'action_low: ' + str(action_low))

    cfg = MATD3Config()                                   # 配置模型参数

    # 创建智能体
    agent_n = [MATD3(ob_dim_n, action_dim_n, state_dim, cfg, agent_id) for agent_id in range(n)]

    total_step = 0                                       # 记录总步数
    epoch = []                                           # 记录轮数，方便画图
    rewards = []                                         # 记录所有回合的奖励
    ave_rewards = []                                     # 记录平均奖励值
    agent_rewards = [[] for _ in range(n)]               # 记录每个智能体的奖励值
    ma_rewards = []                                      # 记录所有回合的 滑动平均奖励
    max_ave_rew = -float('inf')                          # 后几轮最大平均奖励值

    set_seed(env, cfg.seed)                              # 设置随机种子

    # 统计一下运行时间
    starttime = datetime.datetime.now()
    print('\n训练开始时间：', starttime)

    # 是否接着上次训练
    start_ep = 0                                                 # 从第start_ep轮开始训练
    if start_ep == 0:
        make_dir(outputs_path + '/img/', outputs_path + '/model/')  # 创建文件夹

        write_txt_file('outputs_path.txt', outputs_path)  # 将模型存放地址写入txt文件

        print('开始训练！')
    else:
        with open('outputs_path.txt', encoding='utf-8') as f:       # 读取模型存放地址
            outputs_path = f.read()

        for agent_id in range(n):                                   # 加载模型
            agent_n[agent_id].load(outputs_path, agent_id + 1)

        total_step = 137500                                         # 接续训练的总步数
        max_ave_rew = -96.8971                                      # 后几轮最大平均奖励值

        print('从第{}轮开始训练！'.format(start_ep))

    for ep in range(start_ep, cfg.train_eps):
        ep_reward = 0
        agent_reward = [0 for _ in range(n)]                        # 每回合每个智能体的奖励值

        ob_n = env.reset()                                          # 重置环境，获得初始状态

        for step in range(cfg.max_step):
            total_step += 1

            action_n = [agent.choose_action(ob, env, agent_name, mode='train') for agent, ob, agent_name in zip(agent_n, ob_n, range(n))]

            next_ob_n, reward_n, done_n, _ = env.step(action_n)

            for i, reward in enumerate(reward_n):
                agent_reward[i] += reward

            ep_reward += sum(reward_n)

            for agent_id in range(n):
                agent_n[agent_id].memory.store_transition(ob_n[agent_id], action_n[agent_id], reward_n[agent_id], next_ob_n[agent_id], done_n[agent_id])        # 存储经验

            if total_step % 100 == 0:                      # 每100步更新一次智能体
                for agent_id in range(n):                  # 每个智能体单独更新
                    agent_n[agent_id].update(agent_n[agent_id].memory, agent_n)

            ob_n = next_ob_n

            if all(done_n):
                break

        for i in range(n):                                                   # 记录每个智能体每回合奖励值
            agent_rewards[i].append(agent_reward[i])
        ep_reward = round(ep_reward, 3)                                      # 单回合奖励保留后三位小数
        rewards.append(ep_reward)                                            # 记录本回合奖励
        if ma_rewards:                                                       # 记录本回合滑动平均奖励
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (ep + 1) % 1000 == 0:
            epoch.append(ep + 1)                                             # 记录轮数，方便画图
            final_ep_ag_rewards = []                                         # 记录每个智能体平均奖励值
            for rew in agent_rewards:
                final_ep_ag_rewards.append(round(np.mean(rew[-1000:]), 2))
            ave_rewards.append(np.mean(rewards[-1000:]))                     # 记录回合平均奖励值
            endtime = datetime.datetime.now()
            print("Episode:%d | Ave_reward:%.2f | Ave_agent_reward:%s | Steps:%d | Time:%s" \
                % (ep + 1, np.mean(rewards[-1000:]), final_ep_ag_rewards, total_step, endtime - starttime))                # 打印结果

            show(outputs_path, env_name, algo_name, epoch, ave_rewards, ma_rewards, mode='train')     # 绘制曲线

        if (ep + 1) >= (start_ep + cfg.ep_r):                                             # 后几轮平均奖励值较大时，保存模型
            if np.mean(rewards[-int(cfg.ep_r):]) > max_ave_rew:
                max_ave_rew = np.mean(rewards[-int(cfg.ep_r):])

                for agent_id in range(n):                                   # 保存模型
                    agent_n[agent_id].save(outputs_path, ep + 1, cfg.train_eps, max_ave_rew, agent_id + 1)

                print('模型已保存！平均奖励值：%.3f' % max_ave_rew)

    print('训练完成！')
    endtime = datetime.datetime.now()

    all_time = endtime - starttime
    print("\n共计用时:", all_time)                                              # 统计训练用时

# 测试
def test():
    print(f'环境：{env_name} | 算法：{algo_name} | 设备：{device}' + '\n')

    # 创建环境
    # env = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=True)
    env = simple_reference_v2.parallel_env(local_ratio=0.5, max_cycles=25, continuous_actions=True)
    env = mpe_wrapper_for_pettingzoo(env, continuous_actions=True)

    n = env.num_agents                                                 # 环境中智能体的数量
    agents = env.agents                                                # 环境中智能体列表
    ob_dim_n = env.obs_shape_n                                         # 每个智能体的观测维度
    action_dim_n = env.act_shape_n                                    # 每个智能体的动作维度
    state_dim = env.state_space.shape[0]                               # 全部智能体的状态维度

    cfg = MATD3Config()  # 设置参数

    # 创建智能体
    agent_n = [MATD3(ob_dim_n, action_dim_n, state_dim, cfg, agent_id) for agent_id in range(n)]

    with open('outputs_path.txt', encoding='utf-8') as f:     # 读取模型存放地址
        outputs_path = f.read()

    # 加载模型
    for agent_id in range(n):
        agent_n[agent_id].load(outputs_path, agent_id + 1)

    epoch = []  # 记录轮数，方便画图
    rewards = []  # 记录所有回合的奖励
    agent_rewards = [[] for _ in range(n)]  # 记录每个智能体的奖励值
    ma_rewards = []  # 记录所有回合的 滑动平均奖励

    set_seed(env, cfg.seed)                              # 设置随机种子

    print('开始测试！\n')
    for ep in range(cfg.test_eps):
        ep_reward = 0
        agent_reward = [0 for _ in range(n)]             # 每回合每个智能体的奖励值

        ob_n = env.reset()

        for step in range(cfg.max_step):
            action_n = [agent.choose_action(ob, env, agent_name, mode='test') for agent, ob, agent_name in zip(agent_n, ob_n, range(n))]

            next_ob_n, reward_n, done_n, _ = env.step(action_n)

            env.render(mode='human')
            time.sleep(0.1)

            for i, reward in enumerate(reward_n):
                agent_reward[i] += reward

            ep_reward += sum(reward_n)

            ob_n = next_ob_n

            if all(done_n):
                break

        for i in range(n):                                                   # 记录每个智能体每回合奖励值
            agent_rewards[i].append(agent_reward[i])
        epoch.append(ep + 1)                                                 # 记录轮数，方便画图
        final_ep_ag_rewards = []                                             # 记录每个智能体平均奖励值
        for rew in agent_rewards:
            final_ep_ag_rewards.append(round(np.mean(rew), 3))
        ep_reward = round(ep_reward, 3)                                      # 单回合奖励保留后三位小数
        rewards.append(ep_reward)                                            # 记录本回合奖励
        if ma_rewards:                                                       # 记录本回合滑动平均奖励
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print("Episode:%s | Reward:%s | Ave_reward:%.3f | Ave_agent_reward:%s" \
              % (ep + 1, ep_reward, np.mean(rewards), final_ep_ag_rewards))  # 打印结果

        if (ep + 1) % 20 == 0:                                               # 每20轮打印曲线图
            show(outputs_path, env_name, algo_name, epoch, rewards, ma_rewards, mode='test')

    print('平均测试奖励值为：', str(np.mean(rewards)))
    print('测试结束！')


if __name__ == "__main__":
    # 训练模型
    train()

    # 测试模型
    # test()


