"""
training strategy for FightingICE game
Reference:
    Game Page: http://www.ice.ci.ritsumei.ac.jp/~ftgaic/index.htm

@Author: Ruoyu Chen
"""

import os
import argparse

from algorithm import SARSA, DeepQLearning
from fightingice_env import FightingiceEnv
from algorithm import DQN_Learning as DDQN


def mkdir(name):
    '''
    Create folder
    '''
    isExists = os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="FightingICE game")
    # general
    parser.add_argument("--RL-method",
                        type=str,
                        default="Duel",
                        choices=["SARSA", "DDQN", "Duel"],
                        help="Which reinforcement learning algorithm to choose for learning.")
    parser.add_argument("--player", type=str,
                        default="LUD",
                        choices=["ZEN", "LUD", "GARNET"],
                        help="Choose which character to fight.")
    parser.add_argument("--savepath", type=str,
                        default="./checkpoint",
                        help="Choose which character to fight.")

    args = parser.parse_args()
    return args


def main(args):
    """
    Main function.
    """
    mkdir(args.savepath)

    env = FightingiceEnv(character=args.player, port=4242)
    env_args = ["--fastmode", "--grey-bg", "--inverted-player", "1", "--mute", "-r", "100"]
    # this mode let two players have infinite hp, their hp in round can be negative
    # you can close the window display functional by using the following mode
    # env_args = ["--fastmode", "--disable-window", "--grey-bg", "--inverted-player", "1", "--mute"]

    ckpt_save_root = os.path.join(
        os.path.join(args.savepath, args.RL_method), args.player
    )
    mkdir(ckpt_save_root)

    if args.RL_method == "SARSA":
        OPT = SARSA(environment=env, environment_args=env_args)
    elif args.RL_method == "DDQN":
        OPT = DDQN(environment=env, environment_args=env_args)
    elif args.RL_method == "Duel":
        OPT = DeepQLearning(environment=env, environment_args=env_args)

    OPT.Iteration(savepath=ckpt_save_root)


if __name__ == "__main__":
    args = parse_args()
    main(args)
