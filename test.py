"""
Test for game

@Author: Ruoyu Chen
"""

import os
import argparse
import numpy as np

from fightingice_env import FightingiceEnv

def parse_args():
    parser = argparse.ArgumentParser(description="FightingICE game")
    # general
    parser.add_argument("--CKPT",
                        type=str,
                        # default = "checkpoint/SARSA/GARNET/weight.npy",
                        default = "checkpoint/SARSA/ZEN/weight.npy",
                        help="Transformer.")
    parser.add_argument("--player", type=str, 
                        default = "ZEN", 
                        choices=["ZEN", "LUD", "GARNET"],
                        help="Choose which character to fight.")
    parser.add_argument("--Rounds",
                        type=int,
                        default = 100,
                        help="How many rounds to evaluate the outcome.")
    
    args = parser.parse_args()
    return args

def epsilon_greedy(obs, Transformer):
        """
        epsilon-greedy策略
        Return:
            动作值索引 [0, numActions)范围
        """
        # 贪婪策略，选择最优动作
        return np.argmax(
            np.dot(Transformer, obs)
        )

def main(args):
    """
    Main function for test.
    """
    env = FightingiceEnv(character=args.player, port=4242)
    env_args = ["--fastmode", "--grey-bg", "--inverted-player", "1", "--mute"]

    Transformer = np.load(args.CKPT)

    Win_num = 0

    for i in range(args.Rounds):
        """
        一些需要注意的变量
            @ obs: 观测
            @ new_obs: 新的观测
            @ reward: 奖励，这貌似格斗游戏已经给了奖励定义
            @ done: 游戏是否结束
            @ info: [own_hp, opp_hp] 一个列表，agent和opponent的血量
        """
        new_obs = env.reset(env_args=env_args)
        reward, done, info = 0, False, None

        while not done:
            act = epsilon_greedy(new_obs, Transformer)
            # TODO: or you can design with your RL algorithm to choose action [act] according to game state [obs]
            new_obs, reward, done, info = env.step(act)

            if not done:
                # TODO: (main part) learn with data (obs, act, reward, new_obs)
                # suggested discount factor value: gamma in [0.9, 0.95]
                pass
            elif info is not None:
                print("round result: own hp {} vs opp hp {}, you {}".format(info[0], info[1],
                                                                            'win' if info[0]>info[1] else 'lose'))
                if info[0]>info[1]: # 如果获胜，保存信息
                    Win_num += 1
                # 记录每一局信息
                with open("test.log",'a') as f:
                    f.write(args.player + " " +  str(info[0]) + " " + str(info[1]) + " " + str(info[0] - info[1])+"\n")       
            else:
                # java terminates unexpectedly
                pass

    print("In the {} games, AI overcomes MctsAi {} times.".format(args.Rounds, Win_num))

if __name__ == "__main__":
    args = parse_args()
    main(args)