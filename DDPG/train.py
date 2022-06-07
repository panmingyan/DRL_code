import numpy as np
import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

from env import OUNoise

def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境：{cfg.env_name}，算法：{cfg.algo_name}，设备：{cfg.device}')
    ou_noise = OUNoise(env.action_space)  # 动作噪声
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ou_noise.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            action = ou_noise.get_action(action, i_step) 
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()                           # 单步更新
            state = next_state
            # if i_step > cfg.max_step:
            #     done = True
        rewards.append(ep_reward)
        if (i_ep+1)%50 == 0:
            print('回合：{}/{}，奖励：{:.2f}'.format(i_ep+1, cfg.train_eps, np.mean(rewards[-50:])))
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成训练！')
    return rewards, ma_rewards

def test(cfg, env, agent):
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.test_eps):
        state = env.reset() 
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            env.render()
            i_step += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
            # if i_step == cfg.max_step:
            #     done = True
        print('回合：{}/{}, 奖励：{}'.format(i_ep+1, cfg.test_eps, ep_reward))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.1f}")
    ave_rewards = np.mean(rewards)
    print('平均奖励：', ave_rewards)
    print('完成测试！')
    return rewards, ma_rewards