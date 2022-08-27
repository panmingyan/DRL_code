# import random
import numpy as np
# import torch
# import time

from pettingzoo.mpe import simple_v2, simple_spread_v2, simple_adversary_v2, simple_reference_v2, simple_tag_v2
from Wrapper import mpe_wrapper_for_pettingzoo
from pettingzoo.utils import random_demo

# env = simple_spread_v2.env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=True)
# env = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=True)
# env = simple_reference_v2.env(local_ratio=0.5, max_cycles=25, continuous_actions=True)
env = simple_tag_v2.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=True)
env.reset()
# env = mpe_wrapper_for_pettingzoo(env, continuous_actions=True)

# random_demo(env, render=True, episodes=100)




n = env.num_agents                                                # 环境中智能体的数量
agents = env.agents                                                # 环境中智能体列表
# # agent_selection = env.agent_selection                             # 当前智能体
# ob_dim = [env.observation_space(i).shape[0] for i in agents]         # 其中一个智能体的状态维度
# action_dim = [env.action_space(i).shape[0] for i in agents]           # 其中一个智能体的动作维度
# state_dim = env.state_space.shape[0]                              # 全部智能体的状态维度
# ob_type = env.observation_space                                   # 其中一个智能体的状态类型
# action_type = env.action_space                # 其中一个智能体的状态类型
#
# ob = list(env.observation_spaces.values())
#
# # action_random = env.action_space('agent_2').sample()
# #
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        # 检测GPU
# # device_num = torch.cuda.device_count()
# print(ob_type)
# print(ob)

a = [0, 1, 2, 3, 4, 5, 6]
# a = np.array(a)
side_index = [0, 3]
for i, a_num in enumerate(np.array(a)[side_index]):
    print(a_num)
    # print(a[side_index][1])

# ob_n = []
# for i in range(n):
#
#     ob = env.observe(agents[i])
#     ob_n.append(np.array(ob))
#
#     print(ob)
#
# print(ob_n)

# print(ob_n[0])


# for agent in env.agent_iter():
#     observation, reward, done, info = env.last()
#     action = policy(observation, agent)
#     env.step(action)

