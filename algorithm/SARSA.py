"""
SARSA for game

@Author: Ruoyu Chen
"""

import numpy as np
import os
import random

class SARSA():
    """
    SARSA算法

    Data 2022.05.25
    """
    def __init__(self, 
        environment,
        environment_args,
        numStates = 144, 
        numActions = 40, 
        decay = 0.9995, 
        epsilon = 1, 
        gamma = 0.95):
        super(SARSA, self).__init__()
        self.decay = decay      # 衰减
        self.epsilon = epsilon  # 贪婪策略
        self.gamma = gamma      # 折扣因子
        
        self.alpha = 0.01          # 学习率有关
        self.epsilon_decay = 0.95

        self.env = environment  # 环境
        self.env_args = environment_args    # 环境参数

        self.numStates = numStates
        self.numActions = numActions

        # 由于空间是连续或者离散的，为了简化，使用类似函数方式，映射observations到actions，作为价值矩阵。
        self.transformer = np.zeros((self.numActions, self.numStates))

    def epsilon_greedy(self, obs):
        """
        epsilon-greedy策略
        Return:
            动作值索引 [0, numActions)范围
        """
        if np.random.random() > self.epsilon:
            # 贪婪策略，选择最优动作
            return np.argmax(
                np.dot(self.transformer, obs)
            )
            
        else:
            # 随机选择一个动作，打破僵局，优先考虑放大招动作？
            return random.randint(0, self.numActions - 1)
    
    def epsilon_decay_step(self):
        # 最早可以随机选择动作，但是后面需要谨慎选择动作，下限是0.10概率
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.1)

    def Iteration(self, savepath, epochs=500):
        """
        迭代
        """
        Best = 0
        for epoch in range(epochs):
            # 循环训练，每轮游戏状态将初始化，可以获得最开始的obs
            obs = self.env.reset(env_args=self.env_args)
            # 初始化奖励值等
            reward, done, info = 0, False, None
            """
            一些需要注意的变量
                @ obs: 观测
                @ new_obs: 新的观测
                @ reward: 奖励，这貌似格斗游戏已经给了奖励定义
                @ done: 游戏是否结束
                @ info: [own_hp, opp_hp] 一个列表，agent和opponent的血量
            """
            while not done:
                # TODO: or you can design with your RL algorithm to choose action [act] according to game state [obs]
                act = self.epsilon_greedy(obs)

                # 采取策略打拳皇
                new_obs, reward, done, info = self.env.step(act)

                if not done:
                    # TODO: (main part) learn with data (obs, act, reward, new_obs)
                    # 贪心策略
                    new_act = self.epsilon_greedy(new_obs)

                    ## Q learning，利用transformer表示该状态下第A个动作的Q值
                    delta = reward\
                            + self.gamma * np.dot(new_obs, self.transformer[new_act])\
                            - np.dot(obs, self.transformer[act])
                    # 策略提升
                    self.transformer[act] += self.alpha * delta * obs

                    # 更新状态
                    obs = new_obs

                    # 衰减
                    self.epsilon_decay_step()
                    
                elif info is not None:
                    print("round result: own hp {} vs opp hp {}, you {}".format(info[0], info[1],
                                                                                'win' if info[0]>info[1] else 'lose'))
                    if info[0]>info[1]: # 如果获胜，保存信息
                        np.save(
                            os.path.join(savepath, "ckpt-" + str(epoch) + ".npy"), 
                            self.transformer)
                        if info[0] - info[1] > Best:
                            Best = info[0] - info[1]
                            np.save(
                            os.path.join(savepath, "ckpt.npy"), 
                            self.transformer)
                    elif epoch % 10 == 0:
                        np.save(
                            os.path.join(savepath, "temp-" + str(epoch) + ".npy"), 
                            self.transformer)
                else:
                    # java terminates unexpectedly
                    pass
        print("finish training")

            