"""
Minimax Q learning for game

@Author: Ruoyu Chen

Reference:
[1] Littman, Michael L. "Markov games as a framework for multi-agent reinforcement learning." Machine learning proceedings 1994. Morgan Kaufmann, 1994. 157-163.
[2] Zhu, Yuanheng, and Dongbin Zhao. "Online minimax Q network learning for two-player zero-sum Markov games." IEEE Transactions on Neural Networks and Learning Systems (2020).
[3] https://github.com/theevann/MinimaxQ-Learning
"""

import os
import random
import numpy as np
from scipy.optimize import linprog

class MinimaxQ:
    """
    Mini-max Q learning

    Data 2022.05.26
    """
    def __init__(self, 
        environment,
        environment_args,
        numStates = 144, 
        numActionsA = 40,   # 我们可操控的只有40维度
        numActionsB = 56,   # 但敌人就有56维度可以用
        decay = 0.9995, 
        epsilon = 1, 
        gamma = 0.95):
        """
        @ numStates: the dimension of the state space
        @ numActionsA: the dimension of the agent's action space
        @ numActionsB: the dimension of the opponent player's action space
        @ gamma: discounted factor [0, 1], suggest 0.9 or 0.95
        """
        super(MinimaxQ, self).__init__()
        self.decay = decay      # 衰减
        self.epsilon = epsilon  # 贪婪策略
        self.gamma = gamma      # 折扣因子

        self.alpha = 0.1    # 学习率有关

        self.env = environment  # 环境
        self.env_args = environment_args    # 环境参数

        self.numStates = numStates
        self.numActions = numActionsA

        # 由于空间是连续或者离散的，为了简化，使用类似函数方式，映射observations到actions，作为价值矩阵。
        self.transformer = np.ones((self.numActions, self.numStates))

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
        # 最早可以随机选择动作，但是后面需要谨慎选择动作，下限是0.05概率
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.05)

    def Iteraction(self, savepath, epochs=500):
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
                    # Q提升
                    self.transformer[act] += self.alpha * delta * obs

                    # 更新状态
                    obs = new_obs

                    pass
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
                else:
                    # java terminates unexpectedly
                    pass
        print("finish training")

class MinimaxQPlayer:

    def __init__(self, 
        numStates=144, 
        numActionsA=40, 
        numActionsB=56, 
        decay=0.9995, 
        expl=1, 
        gamma=0.95):
        self.decay = decay  # 衰减
        self.expl = expl    # 贪婪策略
        self.gamma = gamma  # 折扣因子
        self.alpha = 1      # 贝尔曼方程超参有关
        self.V = np.ones(numStates) # 价值
        self.Q = np.ones((numStates, numActionsA, numActionsB)) # Q，策略空间
        self.pi = np.ones((numStates, numActionsA)) / numActionsA   # 动作选择概率
        self.numStates = numStates
        self.numActionsA = numActionsA
        self.numActionsB = numActionsB
        self.learning = True

    def chooseAction(self, state, restrict=None):
        """
        贪婪算法选择动作
        """
        if self.learning and np.random.rand() < self.expl:
            action = np.random.randint(self.numActionsA)
        else:
            action = self.weightedActionChoice(state)
        return action

    def weightedActionChoice(self, state):
        rand = np.random.rand()
        cumSumProb = np.cumsum(self.pi[state])
        action = 0
        while rand > cumSumProb[action]:
            action += 1
        return action

    def getReward(self, initialState, finalState, actions, reward, restrictActions=None):
        """
        计算奖励值
        """
        if not self.learning:
            return
        actionA, actionB = actions
        self.Q[initialState, actionA, actionB] = (1 - self.alpha) * self.Q[initialState, actionA, actionB] + \
            self.alpha * (reward + self.gamma * self.V[finalState])
        self.V[initialState] = self.updatePolicy(initialState)  # EQUIVALENT TO : min(np.sum(self.Q[initialState].T * self.pi[initialState], axis=1))
        self.alpha *= self.decay

    def updatePolicy(self, state, retry=False):
        """
        更新策略
        """
        c = np.zeros(self.numActionsA + 1)  # shape (41)
        c[0] = -1
        A_ub = np.ones((self.numActionsB, self.numActionsA + 1))
        A_ub[:, 1:] = -self.Q[state].T  # shape (56, 40)

        b_ub = np.zeros(self.numActionsB)           # shape (56)
        A_eq = np.ones((1, self.numActionsA + 1))   # shape (1, 41)
        A_eq[0, 0] = 0
        b_eq = [1]                                  # shape (1)
        bounds = ((None, None),) + ((0, 1),) * self.numActionsA
        # print(bounds)

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        print(res)

        if res.success:
            self.pi[state] = res.x[1:]  # 40维
        elif not retry:
            return self.updatePolicy(state, retry=True)
        else:
            print("Alert : %s" % res.message)
            return self.V[state]

        return res.x[0]

    def policyForState(self, state):
        for i in range(self.numActionsA):
            print("Actions %d : %f" % (i, self.pi[state, i]))


if __name__ == '__main__':

    def testUpdatePolicy():
        m = MinimaxQPlayer()
        # m.Q[0] = [[0, 1], [1, 0.5]]
        m.updatePolicy(0)
        # print(m.pi)

    testUpdatePolicy()
