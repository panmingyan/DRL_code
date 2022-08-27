import torch
import numpy as np


class ReplayBuffer(object):
    def __init__(self, cfg, ob_dim, action_dim):
        # self.N = agent_num                                 # 智能体数量
        self.buffer_size = cfg.memory_capacity
        self.batch_size = cfg.batch_size
        self.count = 0
        self.current_size = 0
        # self.buffer_obs_n, self.buffer_a_n, self.buffer_r_n, self.buffer_s_next_n, self.buffer_done_n = [], [], [], [], []
        self.buffer_obs, self.buffer_a, self.buffer_r, self.buffer_s_next, self.buffer_done = [], [], [], [], []
        # for agent_id in range(self.N):
        #     self.buffer_obs_n.append(np.empty((self.buffer_size, ob_dim_n[agent_id])))
        #     self.buffer_a_n.append(np.empty((self.buffer_size, action_dim_n[agent_id])))
        #     self.buffer_r_n.append(np.empty((self.buffer_size, 1)))
        #     self.buffer_s_next_n.append(np.empty((self.buffer_size, ob_dim_n[agent_id])))
        #     self.buffer_done_n.append(np.empty((self.buffer_size, 1)))
        self.buffer_obs = np.empty((self.buffer_size, ob_dim))
        self.buffer_a = np.empty((self.buffer_size, action_dim))
        self.buffer_r = np.empty((self.buffer_size, 1))
        self.buffer_s_next = np.empty((self.buffer_size, ob_dim))
        self.buffer_done = np.empty((self.buffer_size, 1))


    def store_transition(self, obs, a, r, obs_next, done):
        self.buffer_obs[self.count] = obs
        self.buffer_a[self.count] = a
        self.buffer_r[self.count] = r
        self.buffer_s_next[self.count] = obs_next
        self.buffer_done[self.count] = done

        # self.buffer_obs.append(obs)
        # self.buffer_a.append(a)
        # self.buffer_r.append(r)
        # self.buffer_s_next.append(obs_next)
        # self.buffer_done.append(done)

        self.count = (self.count + 1) % self.buffer_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def make_index(self):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        return index

    def sample(self, index, device):
        # index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        # batch_obs, batch_a, batch_r, batch_obs_next, batch_done = [], [], [], [], []

        batch_obs = torch.tensor(self.buffer_obs[index], dtype=torch.float).to(device)
        batch_a = torch.tensor(self.buffer_a[index], dtype=torch.float).to(device)
        batch_r = torch.tensor(self.buffer_r[index], dtype=torch.float).to(device)
        batch_obs_next = torch.tensor(self.buffer_s_next[index], dtype=torch.float).to(device)
        batch_done = torch.tensor(self.buffer_done[index], dtype=torch.float).to(device)

        return batch_obs, batch_a, batch_r, batch_obs_next, batch_done