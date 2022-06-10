"""
SARSA for game

@Author: Ruoyu Chen
"""

from multiprocessing.connection import deliver_challenge
import numpy as np
import os
import random
import torch
from torch import nn, optim

import collections

class DuelingQNetwork(nn.Module):
    """
    Implementation of dueling Q-network

    Reference:
    [1] Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." International conference on machine learning. PMLR, 2016.
    """
    def __init__(self, input_dim = 144, output_dim = 40):
        super(DuelingQNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.single_stream = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.ValueNet = nn.Linear(256, 1)
        self.ActionNet = nn.Linear(256, self.output_dim)

    def forward(self, input):
        """
        Forward function
        @ input: shape torch.Size([batch, input_dim])

        Return:
            @ Q_values: shape torch.Size([batch, output_dim])
        """
        # First input into the single-stream network
        feature = self.single_stream(input)

        # Two-stream
        ## state-value
        state_value = self.ValueNet(feature)        # Tensor shape [batch, 1]
        ## advantages for each action
        advantages = self.ActionNet(feature)   # Tensor shape [batch, output_dim]

        # Q-values for each action, please refer the Eq.9 in orginal paper
        Q_values = state_value + advantages - torch.mean(advantages)    # Tensor shape [batch, output_dim]

        return Q_values

class DeepQLearning():
    def __init__(self, 
        environment,
        environment_args,
        numStates = 144, 
        numActions = 40,
        decay = 0.9995, 
        epsilon = 1,
        gamma = 0.95,
        learning_rate = 1e-3,
        batch_size = 32,
        device = 'cuda',
        pretrain = None
    ):
        super(DeepQLearning, self).__init__()
        self.numStates = numStates
        self.numActions = numActions

        # Reinforcement Learning parameters
        self.decay = decay      # 衰减
        self.epsilon = epsilon  # 贪婪策略
        self.gamma = gamma      # 折扣因子

        self.epsilon_decay = 0.95

        self.env = environment              # 环境
        self.env_args = environment_args    # 环境参数

        self.device = device
        
        # This is the learned network DQN
        if pretrain is not None and os.path.exists(pretrain):
            # Init from pretrained model
            self.DQN = torch.load(pretrain, map_location='cpu')
            self.DQN.to(self.device)
            print("Success load pretrained model from {}".format(pretrain))
        else:
            # Init from scratch
            self.DQN = DuelingQNetwork(
                input_dim = self.numStates, 
                output_dim = self.numActions
            ).to(self.device)

        # This is the target-network
        self.TargetNet = DuelingQNetwork(
            input_dim = self.numStates, 
            output_dim = self.numActions
        ).to(self.device)
        self.TargetNet.load_state_dict(self.DQN.state_dict())
        self.TargetNet.eval()

        # Training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.opt = optim.Adam(self.DQN.parameters(), lr = self.learning_rate)
        self.loss = nn.MSELoss()
        self.training = False
        self.training_step = 0
        # self.begin_step = 100

        # Memory Bank，多的自动去除掉
        self.MemoryBank = collections.deque(maxlen=30000)
        # the frequency of updating the target net
        self.update = 0
        self.update_step = 200  

    def epsilon_greedy(self, obs):
        """
        epsilon-greedy策略
        Return:
            动作值索引 [0, numActions)范围
        """
        if np.random.random() > self.epsilon:
            # 贪婪策略，选择最优动作，int()
            return torch.argmax(
                self.DQN(
                    torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                )
            ).item()
            
        else:
            # 随机选择一个动作，打破僵局，优先考虑放大招动作？
            return random.randint(0, self.numActions - 1)
    
    def epsilon_decay_step(self):
        # 最早可以随机选择动作，但是后面需要谨慎选择动作，下限是0.01概率
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)

    def Memorize(self, obs, act, reward, new_obs):
        '''
        Insert the "memory" into buffer.
        The tuple "memory" consists of four parts: (s,a,r,s').
        '''
        self.MemoryBank.append([obs, act, reward, new_obs])
    
    def stepping(self):
        """
        Training step
        """
        self.training_step += 1
        if self.training_step > self.batch_size:
            self.training = True
    
    def sample(self):
        """
        Sample training data from the memory bank
        """
        container_obs = []; container_act = []; container_reward = []; container_new_obs = []

        # 采样batch size个数据
        dataloader = random.sample(self.MemoryBank, self.batch_size)
        # 整理数据
        for data in dataloader:
            obs, act, reward, new_obs = data
            container_obs.append(obs)
            container_act.append([act])
            container_reward.append([reward])
            container_new_obs.append(new_obs)

        container_obs = torch.FloatTensor(container_obs).to(self.device)
        container_act = torch.LongTensor(container_act).to(self.device)
        container_reward = torch.FloatTensor(container_reward).to(self.device)
        container_new_obs = torch.FloatTensor(container_new_obs).to(self.device)

        return container_obs, container_act, container_reward, container_new_obs

    def update_target(self):
        """
        Update the targe network
        """
        self.update +=1
        if self.update == self.update_step:
            self.update = 0
            self.TargetNet.load_state_dict(self.DQN.state_dict())
            self.TargetNet.eval()

    def train(self):
        """
        Training the network
        """
        # Sample data
        container_obs, container_act, container_reward, container_new_obs = self.sample()
        # Q value
        Q_value = self.DQN(container_obs).gather(1, container_act)

        # Next Action
        A_greedy = torch.argmax(self.DQN(container_new_obs), dim = 1, keepdim = True)

        # Target network
        with torch.no_grad():  # calculate target value
            Q_t = self.TargetNet(container_obs).gather(1, A_greedy)
            Q_pi = container_reward + self.gamma * Q_t

        Loss = self.loss(Q_pi, Q_value)
        self.opt.zero_grad()
        Loss.backward()
        self.opt.step()
        return Loss.item()

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
                    # 存储经验
                    self.Memorize(obs, act, reward, new_obs)
                    self.stepping()
                    
                    # 如果可以开始训练网络了
                    if self.training:
                        loss = self.train()
                        with open(os.path.join(savepath, "loss.log"),'a') as f:
                            f.write(str(epoch) + " " + str(loss) +"\n")
                    
                    # 更新状态
                    obs = new_obs

                    # 衰减
                    self.epsilon_decay_step()

                elif info is not None:
                    print("round result: own hp {} vs opp hp {}, you {}".format(info[0], info[1],
                                                                                'win' if info[0]>info[1] else 'lose'))
                    
                    with open(os.path.join(savepath, "train.log"),'a') as f:
                        f.write(str(epoch) + " " +  str(info[0]) + " " + str(info[1]) + " " + str(info[0] - info[1])+"\n")

                    if info[0]>info[1]: # 如果获胜，保存信息
                        torch.save(
                            self.DQN,
                            os.path.join(savepath, "ckpt-" + str(epoch) + ".pt")
                        )
                        
                        if info[0] - info[1] > Best:
                            Best = info[0] - info[1]
                            torch.save(
                                self.DQN,
                                os.path.join(savepath, "ckpt.pt")
                            )

                    elif epoch % 10 == 0:
                        torch.save(
                            self.DQN,
                            os.path.join(savepath, "temp-" + str(epoch) + ".pt")
                        )

                else:
                    # java terminates unexpectedly
                    pass
        print("finish training")