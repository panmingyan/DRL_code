import numpy as np
import gym

from gym import spaces

# 包装器
class mpe_wrapper_for_pettingzoo(gym.Wrapper):
    def __init__(self, env=None, continuous_actions=False):
        gym.Wrapper.__init__(self, env)
        env.reset(seed=666)
        self.continuous_actions = continuous_actions
        self.agents = env.agents
        self.observation_space = [env.observation_space(i) for i in self.agents]
        self.action_space = [env.action_space(i) for i in self.agents]
        assert len(self.observation_space) == len(self.action_space)
        self.num_agents = env.num_agents
        self.obs_shape_n = [self.observation_space[i].shape[0] for i in range(self.num_agents)]
        self.act_shape_n = [self.action_space[i].shape[0] for i in range(self.num_agents)]

    def reset(self):
        obs = self.env.reset()
        return list(obs.values())

    def step(self, actions):
        actions_dict = dict()
        for i, act in enumerate(actions):
            agent = self.agents[i]
            if self.continuous_actions:
                assert np.all(((act<=1.0 + 1e-3), (act>=-1.0 - 1e-3))), \
                    'the action should be in range [-1.0, 1.0], but got {}'.format(act)
                high = self.action_space[i].high
                low = self.action_space[i].low
                mapped_action = low + (act - (-1.0)) * ((high - low) / 2.0)         # [-1, 1] -> [0, 1]
                mapped_action = np.clip(mapped_action, low, high)
                actions_dict[agent] = mapped_action
            else:
                actions_dict[agent] = np.argmax(act)
        obs, reward, done, info = self.env.step(actions_dict)
        return list(obs.values()), list(reward.values()), list(
            done.values()), list(info.values())

