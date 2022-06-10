import random
from fightingice_env import FightingiceEnv

if __name__ == '__main__':
    env = FightingiceEnv(port=4242)
    # for windows user, port parameter is necessary because port_for library does not work in windows
    # for linux user, you can omit port parameter, just let env = FightingiceEnv()

    env_args = ["--fastmode", "--grey-bg", "--inverted-player", "1", "--mute"]
    # this mode let two players have infinite hp, their hp in round can be negative
    # you can close the window display functional by using the following mode
    # env_args = ["--fastmode", "--disable-window", "--grey-bg", "--inverted-player", "1", "--mute"]

    while True:
        """
        一些需要注意的变量
            @ obs: 观测
            @ new_obs: 新的观测
            @ reward: 奖励，这貌似格斗游戏已经给了奖励定义
            @ done: 游戏是否结束
            @ info: [own_hp, opp_hp] 一个列表，agent和opponent的血量
        """
        obs = env.reset(env_args=env_args)
        reward, done, info = 0, False, None

        while not done:
            act = random.randint(0, 39)
            # TODO: or you can design with your RL algorithm to choose action [act] according to game state [obs]
            new_obs, reward, done, info = env.step(act)

            if not done:
                # TODO: (main part) learn with data (obs, act, reward, new_obs)
                # suggested discount factor value: gamma in [0.9, 0.95]
                print(new_obs[0], new_obs[1])
                pass
            elif info is not None:
                print("round result: own hp {} vs opp hp {}, you {}".format(info[0], info[1],
                                                                            'win' if info[0]>info[1] else 'lose'))
            else:
                # java terminates unexpectedly
                pass

    print("finish training")
