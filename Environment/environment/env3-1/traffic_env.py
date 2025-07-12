from __future__ import absolute_import
from __future__ import print_function

import gymnasium as gym
import numpy as np
import os
import sys
import math

import torch
from gymnasium import spaces
import time

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci

gui = True
if gui:
    sumoBinary = checkBinary('sumo-gui')
else:
    sumoBinary = checkBinary('sumo')
config_path = os.path.dirname(__file__)+"/../../../Environment/environment/env3-1/Intersection_3.sumocfg"  # Unprotected left turn in mixed traffic


os.environ['LIBSUMO_AS_TRACI'] = '1'

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ



class Traffic_Env(gym.Env):
    CONNECTION_LABEL = 0  # For traci multi-client support
    def __init__(self, attack=False, adv_steps=2, eval=False):
        self.state_dim = 26
        self.action_dim = 1
        self.maxDistance = 200.0
        self.maxSpeed = 15.0
        self.max_angle = 360.0
        self.max_acc = 7.6
        self.AutoCarID = 'Auto'
        self.reset_times = 0

        # only work for attack
        self.adv_steps = adv_steps
        self.attack_remain = adv_steps
        self.attack = attack
        self.eval = eval
        # For traci multi-client support
        self.label = str(Traffic_Env.CONNECTION_LABEL)
        Traffic_Env.CONNECTION_LABEL += 1
        self.sumo_seed = 0
        # define dims of action space
        # if self.attack:
        #     self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        #     # define dims of state space
        #     self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
        # else:
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        # define dims of state space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

    def raw_obs(self, vehicle_params):

        obs = []
        if self.AutoCarID in vehicle_params:

            zone = [[[],[],[]] for _ in range(6)]
            # t0 = time.perf_counter()
            ego_veh_x, ego_veh_y = traci.vehicle.getPosition(self.AutoCarID)
            # t1 = time.perf_counter()
            # print("vehicle_params: ", vehicle_params)
            # print(f"[Timing] t1-t0: {(t1 - t0) * 1000:.3f} ms, ")
            for VehID in vehicle_params:
                # t2 = time.perf_counter()
                veh_x, veh_y = traci.vehicle.getPosition(VehID)  # position, X & Y
                # t3 = time.perf_counter()
                # print(f"[Timing] t3-t2: {(t3 - t2) * 1000:.3f} ms, ")
                dis = np.linalg.norm(np.array([veh_x-ego_veh_x, veh_y-ego_veh_y]))

                if VehID != self.AutoCarID and dis < self.maxDistance:
                    angle = math.degrees(math.atan2(veh_y-ego_veh_y, veh_x-ego_veh_x))

                    if 0 <= angle < math.degrees(math.atan2(3**0.5, 1)): # 0~60
                        zone[0][0].append(VehID)
                        zone[0][1].append(dis)
                        zone[0][2].append(angle)
                    elif math.degrees(math.atan2(3**0.5, 1)) <= angle < math.degrees(math.atan2(3**0.5, -1)): # 60~120
                        zone[1][0].append(VehID)
                        zone[1][1].append(dis)
                        zone[1][2].append(angle)
                    elif math.degrees(math.atan2(3**0.5, -1)) <= angle < 180: # 120~180
                        zone[2][0].append(VehID)
                        zone[2][1].append(dis)
                        zone[2][2].append(angle)
                    elif -180 <= angle < math.degrees(math.atan2(-3**0.5, -1)): # -180~-120
                        zone[3][0].append(VehID)
                        zone[3][1].append(dis)
                        zone[3][2].append(angle)
                    elif math.degrees(math.atan2(-3**0.5, -1)) <= angle < math.degrees(math.atan2(-3**0.5, 1)): # -120~-60
                        zone[4][0].append(VehID)
                        zone[4][1].append(dis)
                        zone[4][2].append(angle)
                    else: # -60~0
                        zone[5][0].append(VehID)
                        zone[5][1].append(dis)
                        zone[5][2].append(angle)

            for z in zone:
                if len(z[0]) == 0:
                    obs.append(self.maxDistance)
                    obs.append(0.0)
                    obs.append(0.0)
                    obs.append(0.0)
                else:
                    mindis_index = z[1].index(min(z[1]))
                    obs.append(min(z[1]))
                    obs.append(z[2][mindis_index])
                    obs.append(traci.vehicle.getSpeed(z[0][mindis_index]))
                    obs.append(traci.vehicle.getAngle(z[0][mindis_index]))

            obs.append(traci.vehicle.getSpeed(self.AutoCarID))
            obs.append(traci.vehicle.getAngle(self.AutoCarID))
            info = {'x_position': ego_veh_x, 'y_position': ego_veh_y, 'reward': 0.0, 'cost': 0.0, 'flag': False}
        else:
            obs = [self.maxDistance, 0.0, 0.0, 0.0, self.maxDistance, 0.0, 0.0, 0.0, self.maxDistance, 0.0, 0.0, 0.0,\
                   self.maxDistance, 0.0, 0.0, 0.0,self.maxDistance, 0.0, 0.0, 0.0, self.maxDistance, 0.0, 0.0, 0.0,\
                   0.0, 0.0]
            info = {'x_position': 0.0, 'y_position': 0.0, 'reward': 0.0, 'cost': 0.0, 'flag': False}


        return obs, info

    def obs_to_state(self, vehicle_params):
        # t0 = time.perf_counter()
        obs, info = self.raw_obs(vehicle_params)
        # t1 = time.perf_counter()
        # print("raw_obs===>", obs)
        state = [obs[0]/self.maxDistance, obs[1]/self.max_angle, obs[2]/self.maxSpeed, obs[3]/self.max_angle,\
                 obs[4]/self.maxDistance, obs[5]/self.max_angle, obs[6]/self.maxSpeed, obs[7]/self.max_angle,\
                 obs[8]/self.maxDistance, obs[9]/self.max_angle, obs[10]/self.maxSpeed, obs[11]/self.max_angle,\
                 obs[12]/self.maxDistance, obs[13]/self.max_angle, obs[14]/self.maxSpeed, obs[15]/self.max_angle,\
                 obs[16]/self.maxDistance, obs[17]/self.max_angle, obs[18]/self.maxSpeed, obs[19]/self.max_angle,\
                 obs[20]/self.maxDistance, obs[21]/self.max_angle, obs[22]/self.maxSpeed, obs[23]/self.max_angle,\
                 obs[24]/self.maxSpeed, obs[25]/self.max_angle]
        # t2 = time.perf_counter()
        # if self.attack:
        #     state.append(self.attack_remain/self.adv_steps)
        #     state.append(0)
        # print(f"[Timing] raw_obs: {(t1 - t0) * 1000:.3f} ms, "
        #       f"state: {(t2 - t1) * 1000:.3f} ms, ")
        return state, info

    def get_reward_a(self, vehicle_params):
        cost = 0.0
        # t0 = time.perf_counter()
        raw_obs, _ = self.raw_obs(vehicle_params)
        # t1 = time.perf_counter()

        dis_fr = raw_obs[0]
        dis_f = raw_obs[4]
        dis_fl = raw_obs[8]
        dis_rl = raw_obs[12]
        dis_r = raw_obs[16]
        dis_rr = raw_obs[20]
        v_ego = raw_obs[24]
        dis_sides = [dis_fr, dis_fl, dis_rl, dis_rr]

        # t2 = time.perf_counter()
        # efficiency
        reward = v_ego / self.maxSpeed

        # t3 = time.perf_counter()
        # safety
        collision_value = self.check_collision(dis_f, dis_r, dis_sides, vehicle_params)
        # t4 = time.perf_counter()
        if collision_value is True:
            cost = 1.0

        # print(f"[Timing] raw_obs: {(t1 - t0) * 1000:.3f} ms, "
        #       f"dis_sides: {(t2 - t1) * 1000:.3f} ms, "
        #       f"reward: {(t3 - t2) * 1000:.3f} ms, "
        #       f"check_collision: {(t4 - t3) * 1000:.3f} ms, ")

        return reward-cost, collision_value, reward, cost

    def check_collision(self, dis_f, dis_r, dis_sides, vehicle_params):
        collision_value = False

        if (dis_f < 2.0) or (dis_r < 1.5) or (min(dis_sides) < 1.0):
            collision_value = True
            print("--->Checker-1: Collision!")
        elif self.AutoCarID not in vehicle_params:
            collision_value = True
            print("===>Checker-2: Collision!")

        return collision_value

    def step(self, action_a):

        action = self.max_acc * action_a
        if isinstance(action, np.ndarray):
            # 假设 action 是一个标量数组，比如 array([0])，用 item() 取标量
            action_val = action.item()
        elif isinstance(action, torch.Tensor):
            action_val = action.cpu().item()
        speed = float(traci.vehicle.getSpeed(self.AutoCarID) + action_val)
        traci.vehicle.setSpeed(self.AutoCarID, max(speed, 0.001))

        # traci.vehicle.setSpeed(self.AutoCarID, max(traci.vehicle.getSpeed(self.AutoCarID) + action, 0.001))
        traci.simulationStep()

        # Get the new vehicle parameters
        new_vehicle_params = traci.vehicle.getIDList()
        reward_cost, collision_value, reward, cost = self.get_reward_a(new_vehicle_params)
        next_state, info = self.obs_to_state(new_vehicle_params)

        info['reward'] = reward
        info['cost'] = cost

        if self.attack:
            return np.array(next_state, dtype=np.float32), cost, collision_value, False, info
        else:
            return np.array(next_state, dtype=np.float32), reward_cost, collision_value, False, info

    def reset(self, seed=None, options=None):
        self.attack_remain = self.adv_steps
        if options is None:
            if self.reset_times % 2 == 0:
                self.sumo_seed = "%d" % self.reset_times
        else:
            self.sumo_seed = 'random'
        self.start()

        # traci.load(["-c", config_path])
        print('Resetting the layout!!!!!!', self.reset_times)
        self.reset_times += 1

        AutoCarAvailable = False
        while AutoCarAvailable == False:
            traci.simulationStep()
            VehicleIds = traci.vehicle.getIDList()
            if self.AutoCarID in VehicleIds:
                AutoCarAvailable = True

        # Just check if the auto car still exisits and that there has not been any collision
        for VehId in VehicleIds:
            if VehId == self.AutoCarID:
                traci.vehicle.setSpeedMode(VehId, 22)
                traci.vehicle.setLaneChangeMode(VehId, 0)  # Disable automatic lane changing

        initial_state, info = self.obs_to_state(VehicleIds)

        return np.array(initial_state, dtype=np.float32), info

    def close(self):
        traci.close()



    # def start(self, gui=False):
        # # t0 = time.perf_counter()
        # sumoBinary = checkBinary('sumo-gui') if gui else checkBinary('sumo')
        # # t1 = time.perf_counter()
        # # print(f"[Profiling] checkBinary 耗时: {t1 - t0:.3f}s")
        #
        # sumo_cmd = [sumoBinary, "-c", config_path, "--collision.check-junctions", "true"]
        # t2 = time.perf_counter()
        # print(f"[Profiling] 构造命令列表耗时: {t2 - t1:.3f}s")
        #
        # try:
        #     traci.close()
        # except:
        #     pass
        # t3 = time.perf_counter()
        # print(f"[Profiling] traci.close() 耗时: {t3 - t2:.3f}s")
        #
        # # … 同理给 append/extend、条件分支也打点 …
        #
        # t4 = time.perf_counter()
        # if LIBSUMO:
        #     traci.start(sumo_cmd)
        # else:
        #     traci.start(sumo_cmd, label=self.label)
        # t5 = time.perf_counter()
        # print(f"[Profiling] traci.start() 耗时: {t5 - t4:.3f}s")

        # 后面如果紧接着是模拟步进，也可以继续打点

    def start(self, gui=False):
        sumoBinary = checkBinary('sumo-gui') if gui else checkBinary('sumo')
        sumo_cmd = [sumoBinary, "-c", config_path, "--collision.check-junctions", "true"]
        try:
            traci.close()
        except:
            pass  # 如果没有活跃的连接，忽略异常

        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])

        if LIBSUMO:
            traci.start(sumo_cmd)
        else:
            traci.start(sumo_cmd, label=self.label)
            # traci.start(sumo_cmd)

    def get_obs(self):
        return self.obs

    def set_obs(self, obs):
        self.obs = obs


