import torch
import torch.nn as nn
from torch.nn import ReLU, Linear, MSELoss
import numpy as np
import os
import random
from math import sqrt
from copy import deepcopy

'''

Dueling DQN Algorithm
@ Authors: N. Chen, R. Chen.
Thanks to the contribution of my teammate, Ruoyu Chen!

'''

class DuelDQN(nn.Module):

    def __init__(self, state_dim=23, action_dim=40):
        '''
        The network consists of 2 approximator: ValueNet and ActionNet.
        The final Q function is combined as follows:
        Q(s,a) = ValueNet(s) + ActionNet(s,a) - mean(ActionNet(s,a))
        
        '''
        super(DuelDQN, self).__init__()
        self.linear1 = Linear(state_dim, 500)
        self.linear2 = Linear(500, 500)
        self.ValueOutput = Linear(500, 1)
        self.ActionOutput = Linear(500, action_dim)

    def forward(self, s):
        '''
        The VNet and ANet share the state representation.
        VNet aims to compute V*(s), while ANet aims to calculate A*(s,a)
        '''
        hidden = self.linear1(s)
        hidden = self.linear2(hidden)
        ValueNet = self.ValueOutput(hidden)
        ActionNet = self.ActionOutput(hidden)
        return ValueNet, ActionNet


class DQN_Learning():
    def __init__(self, 
        environment, 
        environment_args, 
        numStates=21, 
        numActions=40, 
        decay=0.995, 
        epsilon=1, 
        gamma=0.95,
        device = 'cuda',
        pretrain = None
        ):

        self.decay = decay      # decay rate
        self.epsilon = epsilon  # exploration rate
        self.gamma = gamma  # discounting rate

        self.device = device

        self.alpha = 0.1  # learning rate
        self.epsilon_decay = 0.95

        self.env = environment  # fighting env
        self.env_args = environment_args  # params of fighting env

        self.numStates = numStates  # number of states
        self.numActions = numActions  # number of actions

        if pretrain is not None and os.path.exists(pretrain):
            self.learnNet = torch.load(pretrain)
            print("Success load pretrained model from {}".format(pretrain))
        else:
            self.learnNet = DuelDQN().to(self.device)  # DuelDQN().to('cuda')  # learning DDQN
        self.targetNet = deepcopy(self.learnNet)  # target DDQN
        self.batch_size = 256  # mini-batch size
        self.update_step = 200  # the frequency of updating the target net
        self.optim = torch.optim.Adam(self.learnNet.parameters())
        self.loss = MSELoss()

        self.buffer = []  # initially the buffer is empty
        self.buffer_index = 0  # next index of buffer
        self.buffer_capacity = 20000  # capacity of buffer

        self.ConvertJavaIntoIndex = [
            25, -1, 23, 21, 15, -1, 24, 22, 14, -1, 37, 20, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 38, 39, -1, -1, 26, 27,
            16, 17, 0, 1, 6, 7, 35, 36, 18, 19, 10, 11, 12, 13, 30, 31, 33,
            34, 28, 29, 4, 5, 8, 9, 2, 3, 32]

    def Memorize(self, s, a, r, s1):
        '''
        Insert the "memory" into buffer.
        The tuple "memory" consists of four parts: (s,a,r,s').
        '''
        memory = {'s': s, 'a': a, 'r': r, 's1': s1}
        if len(self.buffer) < self.buffer_capacity:
            self.buffer_index += 1
            self.buffer.append(memory)
        else:
            self.buffer[self.buffer_index % self.buffer_capacity] = memory
            self.buffer_index += 1

    def Sampling(self):
        '''
        Randomly sample a mini-batch.
        '''
        rand_position = np.random.choice(np.arange(0, len(self.buffer)), self.batch_size, replace=False)
        s = torch.cat([torch.tensor(self.buffer[k]['s']).unsqueeze(0) for k in rand_position], dim=0).to(self.device)
        a = [self.buffer[k]['a'] for k in rand_position]
        r = [self.buffer[k]['r'] for k in rand_position]
        s1 = torch.cat([torch.tensor(self.buffer[k]['s1']).unsqueeze(0) for k in rand_position], dim=0).to(self.device)
        return s, a, r, s1

    def Ready(self):
        '''
        Decide whether the buffer is large enough to sample a mini-batch.
        '''
        return len(self.buffer) >= self.batch_size

    def Analyze_obs(self, obs):
        '''
        split the observation into four parts: 
            state of P1, 
            action of P1, 
            state of P2, 
            action of P2, 
        namely (s1,s2,p1,p2)
        '''
        s1 = np.array([obs[0], obs[1], obs[2], obs[3] - obs[69], obs[4] - obs[70], obs[5], obs[6], obs[7], obs[8], obs[71], obs[72], obs[73], obs[74], obs[138], obs[139], obs[140], obs[141], obs[142], obs[143], obs[65], obs[131], obs[67], obs[68]])
        a1 = self.ConvertJavaIntoIndex[np.argmax(obs[9:65])]
        s2 = np.array([obs[66], obs[67], obs[68], obs[69] - obs[3], obs[70] - obs[4], obs[71], obs[72], obs[73], obs[74], obs[5], obs[6], obs[7], obs[8], obs[132], obs[133], obs[134], obs[135], obs[136], obs[137], obs[131], obs[65], obs[1], obs[2]])
        a2 = self.ConvertJavaIntoIndex[np.argmax(obs[75:131])]
        return s1, a1, s2, a2

    def Decision(self, s, explore=True):
        '''
        The decision process of agent. if explore=True (training phase), adopt epsilon-greedy strategy, otherwise use argmax of Q.
        '''
        Input = torch.tensor(s).to(self.device)
        Value, Action = self.learnNet(Input)
        Value, Action = Value.cpu().detach().numpy(), Action.cpu().detach().numpy()
        Q = Value - np.mean(Action) + Action
        if explore:
            if np.random.random() > self.epsilon:  # exploit
                return np.argmax(Q)
            else:
                return random.randint(0, self.numActions - 1)  # explore
        else:
            return np.argmax(Q)  # fully exploit

    def epsilon_decay_step(self):
        # 最早可以随机选择动作，但是后面需要谨慎选择动作，下限是0.10概率
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)

    def Iteration(self, savepath, epochs=500):
        '''
        Iteration of learning procedure. The most important part of algorithm!
        '''

        for epoch in range(epochs):  # an epoch of training
            obs = self.env.reset(env_args=self.env_args)  # firstly, reset the environment
            reward, done, info = 0, False, None
            update = 0  # when to update the target network

            if len(obs) == 144:
                now_s1, now_a1, now_s2, now_a2 = self.Analyze_obs(obs)
            else:
                continue

            while not done:
                action = self.Decision(now_s1, explore=True)
                new_obs, reward, done, info = self.env.step(action)  # play with the chosen action
                if not done:
                    new_s1, new_a1, new_s2, new_a2 = self.Analyze_obs(new_obs)  # get a "memory" from replay information
                    if now_a1 != -1:
                        self.Memorize(now_s1, now_a1, reward, new_s1)
                    if now_a2 != -1:
                        self.Memorize(now_s2, now_a2, -reward, new_s2)
                    now_s1, now_a1, now_s2, now_a2 = new_s1, new_a1, new_s2, new_a2
                    if self.Ready():  # ready for train DQN
                        s, a, r, s1 = self.Sampling()  # sample a mini-batch of training data
                        Value, Action = self.learnNet(s)  # a forward step
                        Q = Value + Action - torch.mean(Action, dim=-1, keepdim=True)
                        with torch.no_grad():  # calculate target value
                            Value_t, Action_t = self.targetNet(s)
                            Q_t = Value_t + Action_t - torch.mean(Action_t, dim=-1, keepdim=True)
                            target = torch.tensor(r).to(self.device) + self.gamma * torch.max(Q_t, dim=-1)[0].to(self.device)  # target value
                        loss = self.loss(Q[np.arange(self.batch_size), a], target.detach())  # calculate MSE loss
                        self.optim.zero_grad()
                        loss.backward()

                        with open(os.path.join(savepath, "loss.log"),'a') as f:
                            f.write(str(epoch) + " " + str(loss.item()) +"\n")

                        # print('loss: %.3f' % loss)  # It seems that the loss of DQN hardly converges...
                        self.optim.step()
                        update += 1
                        if update == self.update_step:
                            update = 0
                            self.targetNet = deepcopy(self.learnNet)  # update params of target network

                        # 衰减
                        self.epsilon_decay_step()
                        
                elif info is not None:
                    print("round result: own hp {} vs opp hp {}, you {}".format(info[0], info[1], 'win' if info[0] > info[1] else 'lose'))
                    
                    torch.save(self.learnNet, os.path.join(savepath, "cn-ckpt-" + str(epoch) + ".pt"))  # save the model

                    with open(os.path.join(savepath, "train.log"),'a') as f:
                        f.write(str(epoch) + " " +  str(info[0]) + " " + str(info[1]) + " " + str(info[0] - info[1])+"\n")        
        
        print('Done!')

