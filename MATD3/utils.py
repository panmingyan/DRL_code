#!/usr/bin/env python
# coding=utf-8

import gym
from gym import spaces
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from matplotlib.font_manager import FontProperties  # 导入字体模块


# 设置随机种子
def set_seed(env, seed):
    if seed != 0:
        torch.manual_seed(seed)
        # env.reset(seed=seed)
        np.random.seed(seed)

# 创建文件夹
def make_dir(*paths):
    ''' 创建文件夹
    '''
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

# 将地址写入文件
def write_txt_file(file_path, outputs_path):
    f = open(file_path, "w")
    f.write(outputs_path)
    f.close()

# 绘图
def show(outputs_path, env_name, algo_name, epoch, rewards, ma_rewards, mode):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title(env_name, fontproperties=font)     # 显示中文标题
    plt.title(env_name + '_' + algo_name + '_' + mode)
    plt.xlabel('epsiodes')
    plt.ylabel('rewards')
    plt.plot(epoch, rewards)
    if mode == 'test':
        plt.plot(epoch, ma_rewards)
    plt.axhline(y=-57.1, ls="-", c="red", linewidth=0.6)         # 添加水平直线
    plt.savefig(outputs_path + '/img/' + env_name+'_'+algo_name +'_' +mode +'.png')                           # 保存图片
    plt.show()



