'''
Author: P.M.Y
Date: 2022-5-25
'''
import numpy as np
import torch

def train(cfg,env,agent):
    print('开始训练！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    steps = 0
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, prob, val = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            steps += 1
            ep_reward += reward
            agent.memory.push(state, action, prob, val, reward, done)  # 将得到的相关量放入memory
            if steps % cfg.update_fre == 0:
                agent.update()
            state = state_
        if type(ep_reward) == torch.Tensor:   # 如果ep_reward是tensor类型，则取出tensor中的值
            ep_reward = ep_reward.item()
        print('Episode:', (i_ep + 1), ' Reward:', ep_reward)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep+1)%50 == 0:
            # print(f"回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}")
            print("Episode:{}/{}: Ave_Reward:{}".format(i_ep + 1, cfg.train_eps, np.mean(rewards)) + '\n')
    print('完成训练！')
    return rewards,ma_rewards

def eval(cfg,env,agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        # env.render()
        done = False
        ep_reward = 0
        while not done:
            env.render()
            action, prob, val = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            ep_reward += reward
            state = state_
            # env.render()
        if type(ep_reward) == torch.Tensor:   # 如果ep_reward是tensor类型，则取出tensor中的值
            ep_reward = ep_reward.item()
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print('Episode:{}/{}   reward:{}'.format(i_ep+1, cfg.test_eps, ep_reward))
    print('完成训练！')
    return rewards,ma_rewards

if __name__ == '__main__':
    import sys,os
    curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
    parent_path = os.path.dirname(curr_path) # 父路径
    sys.path.append(parent_path) # 添加路径到系统路径

    import gym
    import torch
    import datetime
    from common.plot import plot_rewards
    from common.utils import save_results,make_dir
    from PPO.agent import PPO
    from PPO.train import train

    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间

    class PPOConfig:
        def __init__(self) -> None:
            self.algo = "DQN"  # 算法名称
            self.env_name = 'CartPole-v0' # 环境名称
            self.continuous = False # 环境是否为连续动作
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
            self.train_eps = 200 # 训练的回合数
            self.test_eps = 20 # 测试的回合数
            self.batch_size = 5
            self.gamma=0.99
            self.n_epochs = 4
            self.actor_lr = 0.0003
            self.critic_lr = 0.0003
            self.gae_lambda=0.95
            self.policy_clip=0.2
            self.hidden_dim = 256
            self.update_fre = 20 # frequency of agent update

    class PlotConfig:
        def __init__(self) -> None:
            self.algo = "DQN"  # 算法名称
            self.env_name = 'CartPole-v0' # 环境名称
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
            self.result_path = curr_path+"/outputs/" + self.env_name + \
                '/'+curr_time+'/results/'  # 保存结果的路径
            self.model_path = curr_path+"/outputs/" + self.env_name + \
                '/'+curr_time+'/models/'  # 保存模型的路径
            self.save = True # 是否保存图片

    def env_agent_config(cfg,seed=1):
        env = gym.make(cfg.env_name)  
        env.seed(seed)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = PPO(state_dim,action_dim,cfg)
        return env,agent

    cfg  = PPOConfig()
    plot_cfg = PlotConfig()
    # 训练
    env,agent = env_agent_config(cfg,seed=1)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(plot_cfg.result_path, plot_cfg.model_path) # 创建保存结果和模型路径的文件夹
    agent.save(path=plot_cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path)
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")
    # 测试
    env,agent = env_agent_config(cfg,seed=10)
    agent.load(path=plot_cfg.model_path)
    rewards,ma_rewards = eval(cfg,env,agent)
    save_results(rewards,ma_rewards,tag='eval',path=plot_cfg.result_path)
    plot_rewards(rewards,ma_rewards,plot_cfg,tag="eval")