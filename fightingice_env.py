import os
import platform
import random
import subprocess
import time
import numpy
from multiprocessing import Pipe
from threading import Thread

import gym
from gym import error, spaces, utils
from py4j.java_gateway import (CallbackServerParameters, GatewayParameters,
                               JavaGateway, get_field)

from gym_ai import GymAI


def game_thread(env):
    try:
        env.game_started = True
        env.manager.runGame(env.game_to_start)
    except:
        env.game_started = False
        print("Please IGNORE the Exception above because of restart java game")


class FightingiceEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        

        self.freq_restart_java = 1       # frequency of rounds to restart java env
        self.java_env_path = os.getcwd()

        if "java_env_path" in kwargs.keys():
            self.java_env_path = kwargs["java_env_path"]
        if "freq_restart_java" in kwargs.keys():
            self.freq_restart_java = kwargs["freq_restart_java"]
        if "character" in kwargs.keys():
            self.character = kwargs["character"]
        else:
            self.character = 'ZEN'  # 这里是控制游戏的
        if "port" in kwargs.keys():
            self.port = kwargs["port"]
        else:
            try:
                import port_for
                self.port = port_for.select_random()  # select one random port for java env
            except:
                raise ImportError(
                    "Pass port=[your_port] when make env, or install port_for to set startup port automatically, maybe pip install port_for can help")


        _actions = "AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_UA AIR_UB BACK_JUMP BACK_STEP CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD DASH FOR_JUMP FORWARD_WALK JUMP NEUTRAL STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD THROW_A THROW_B"
        action_strs = _actions.split(" ")    # 40 actions

        self.observation_space = spaces.Box(low=-1, high=1, shape=(144,))
        self.action_space = spaces.Discrete(len(action_strs))

        os_name = platform.system()
        if os_name.startswith("Linux"):
            self.system_name = "linux"
        elif os_name.startswith("Darwin"):
            self.system_name = "macos"
        else:
            self.system_name = "windows"

        if self.system_name == "linux":
            # first check java can be run, can only be used on Linux
            java_version = subprocess.check_output(
                'java -version 2>&1 | awk -F[\\\"_] \'NR==1{print $2}\'', shell=True)
            if java_version == b"\n":
                raise ModuleNotFoundError("Java is not installed")
        else:
            print("Please make sure you can run java if you see some error")

        # second check if FightingIce is installed correct
        start_jar_path = os.path.join(self.java_env_path, "FightingICE.jar")
        start_data_path = os.path.join(self.java_env_path, "data")
        start_lib_path = os.path.join(self.java_env_path, "lib")
        lwjgl_path = os.path.join(start_lib_path, "lwjgl", "*")
        lib_path = os.path.join(start_lib_path, "*")
        start_system_lib_path = os.path.join(
            self.java_env_path, "lib", "natives", self.system_name)
        natives_path = os.path.join(start_system_lib_path, "*")
        if os.path.exists(start_jar_path) and os.path.exists(start_data_path) and os.path.exists(start_lib_path) and os.path.exists(start_system_lib_path):
            pass
        else:
            error_message = "FightingICE is not installed in your script launched path {}, set path when make() or start script in FightingICE path".format(
                self.java_env_path)
            raise FileExistsError(error_message)
        self.java_ai_path = os.path.join(self.java_env_path, "data", "ai")
        ai_path = os.path.join(self.java_ai_path, "*")
        if self.system_name == "windows":
            self.start_up_str = "{};{};{};{};{}".format(
                start_jar_path, lwjgl_path, natives_path, lib_path, ai_path)
            self.need_set_memory_when_start = True
        else:
            self.start_up_str = "{}:{}:{}:{}:{}".format(
                start_jar_path, lwjgl_path, natives_path, lib_path, ai_path)
            self.need_set_memory_when_start = False

        self.game_started = False
        self.round_num = 0
        self.win = False

    def _start_java_game(self, env_args=None):
        # start game
        print("Start java env in {} and port {}".format(
            self.java_env_path, self.port))
        devnull = open(os.devnull, 'w')

        if env_args is None:
            env_args = ["--fastmode", "--grey-bg", "--inverted-player", "1", "--mute", "--limithp", "400", "400"]

        if self.system_name == "windows":
            # -Xms1024m -Xmx1024m we need set this in windows
            self.java_env = subprocess.Popen(["java", "-Xms1024m", "-Xmx1024m", "-cp", self.start_up_str, "Main",
                                              "--port", str(self.port), "--py4j"] + env_args, stdout=devnull, stderr=devnull)
        elif self.system_name == "linux":
            self.java_env = subprocess.Popen(["java", "-cp", self.start_up_str, "Main", "--port", str(self.port),
                                              "--py4j"] + env_args, stdout=devnull, stderr=devnull)
        elif self.system_name == "macos":
            self.java_env = subprocess.Popen(["java", "-XstartOnFirstThread", "-cp", self.start_up_str, "Main",
                                              "--port", str(self.port), "--py4j"] + env_args, stdout=devnull, stderr=devnull)
        # self.java_env = subprocess.Popen(["java", "-cp", "/home/myt/gym-fightingice/gym_fightingice/FightingICE.jar:/home/myt/gym-fightingice/gym_fightingice/lib/lwjgl/*:/home/myt/gym-fightingice/gym_fightingice/lib/natives/linux/*:/home/myt/gym-fightingice/gym_fightingice/lib/*", "Main", "--port", str(self.free_port), "--py4j", "--c1", "ZEN", "--c2", "ZEN","--fastmode", "--grey-bg", "--inverted-player", "1", "--mute"])
        # sleep 3s for java starting, if your machine is slow, make it longer
        time.sleep(3)

    def _start_gateway(self, p2="MctsAi"):
        # auto select callback server port and reset it in java env
        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(
            port=self.port), callback_server_parameters=CallbackServerParameters(port=0))
        python_port = self.gateway.get_callback_server().get_listening_port()
        self.gateway.java_gateway_server.resetCallbackClient(
            self.gateway.java_gateway_server.getCallbackClient().getAddress(), python_port)
        self.manager = self.gateway.entry_point

        # create pipe between gym_env_api and python_ai for java env
        server, client = Pipe()
        self.pipe = server
        self.p1 = GymAI(self.gateway, client)
        self.p1_name = self.p1.__class__.__name__
        self.manager.registerAI(self.p1_name, self.p1)

        if isinstance(p2, str):
            # p2 is a java class name
            self.p2 = p2
            self.p2_name = p2
        else:
            # p2 is a python class
            self.p2 = p2(self.gateway)
            self.p2_name = self.p2.__class__.__name__
            self.manager.registerAI(self.p2_name, self.p2)

        if random.random() > 0.5:
            self.p1, self.p2 = self.p2, self.p1
            self.p1_name, self.p2_name = self.p2_name, self.p1_name
        
        self.game_to_start = self.manager.createGame(self.character, self.character,
                                                     self.p1_name, self.p2_name, self.freq_restart_java)
        print("start fightingice env: {} vs {} in {}".format(self.p1_name, self.p2_name, self.character))

        self.game = Thread(target=game_thread,
                           name="game_thread", args=(self, ))
        self.game.start()

        self.game_started = True
        self.round_num = 0
        self.win = False

    def _close_gateway(self):
        self.gateway.close_callback_server()
        self.gateway.close()
        del self.gateway

    def _close_java_game(self):
        self.java_env.kill()
        del self.java_env
        self.pipe.close()
        del self.pipe
        self.game_started = False
        time.sleep(3)

    def reset(self, p2="MctsAi", env_args=None):
        # start java game if game is not started
        if self.game_started is False:
            try:
                self._close_gateway()
                self._close_java_game()
            except:
                pass
            self._start_java_game(env_args)
            self._start_gateway(p2)

        # to provide crash, restart java game in some freq
        if self.round_num >= self.freq_restart_java:
            try:
                self._close_gateway()
                self._close_java_game()
                self._start_java_game(env_args)
            except:
                raise SystemExit("Can not restart game")
            self._start_gateway(p2)

        # just reset is anything ok
        self.pipe.send("reset")
        self.round_num += 1
        if self.pipe.poll(30):
            # wait for 30s for java reply, if your computer is slow, increase this waiting time
            obs = self.pipe.recv()
        else:
            print("fail in reset and let's do it again.")
            obs = self.reset(p2=p2, env_args=env_args)
        return obs

    def step(self, action):
        # check if game is running, if not try restart
        # when restart, dict will contain crash info, agent should do something, it is a BUG in this version
        if self.game_started is False:
            dict = {}
            dict["pre_game_crashed"] = True
            obs = self.reset()
            return obs, 0, None, dict

        self.pipe.send(["step", action])
        if self.pipe.poll(30):
            new_obs, reward, done, info = self.pipe.recv()
            print("current hp: own {} vs opp {}".format(info[0], info[1]))
            # you can close the print functional to accelerate training speed
        else:
            new_obs, reward, done, info = None, 0, True, None
            print("can't receive signals within 30 seconds. let's terminate gym env.")
        return new_obs, reward, done, info

    def render(self, mode='human'):
        # no need
        pass

    def close(self):
        if self.game_started:
            try:
                self._close_gateway()
            except:
                pass
            self._close_java_game()


