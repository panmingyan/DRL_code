import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import math, os
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000
MAX_EP_STEP = 200                                  # 每回合最大步数

env = gym.make('Pendulum-v1')
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 200)
        self.mu = nn.Linear(200, a_dim)
        self.sigma = nn.Linear(200, a_dim)
        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a1 = F.relu6(self.a1(x))
        mu = 2 * F.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001      # avoid 0
        c1 = F.relu6(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss


res = []                    # record episode reward to plot

a3c_t = Net(N_S, N_A)
a3c_t.load_state_dict(torch.load('./models/A3C_Pendulum_v1_9.pth'))    # 载入模型

for i in range(10):
    s = env.reset()  # 重置环境
    ep_r = 0  # 初始化该循环对应的episode的总奖励
    while True:                                                         # 开始一个episode (每一个循环代表一步)
        env.render()                                                    # 显示实验动画
        a = a3c_t.choose_action(v_wrap(s[None, :]))
        s_,r,done,info=env.step(a.clip(-2, 2))                                # 执行动作，获得反馈
        ep_r += r
        if done:
            print('Ep_r: ', ep_r)
            res.append(ep_r)
            break
        s = s_

plt.plot(res)
plt.ylabel('Moving average ep reward')
plt.xlabel('Step')
plt.show()