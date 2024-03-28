# -*- coding: utf-8 -*-

'''

Interacting with traffic_light_dqn.py and map_computor.py

1) retriving values from sumo_computor.py

2) update state

3) controling logic

'''

from agent import State
from sys import platform
import sys
import os
import map_computor
import numpy as np
import shutil
import json
import collections

from agent import State

import cityflow as engine

class Vehicles:
    initial_speed = 5.0

    def __init__(self):
        # add what ever you need to maintain
        self.id = None
        self.speed = None
        self.wait_time = None
        self.stop_count = None
        self.enter_time = None
        self.has_read = False
        self.first_stop_time = -1
        self.entering = True
        # --- my modification ---
        self.recount_waiting_time = 0


class CityFlowAgent:

    class ParaSet:

        def __init__(self, dic_paras):
            for key, value in dic_paras.items():
                setattr(self, key, value)

            if hasattr(self, "REWARDS_INFO_DICT"):
                self.LIST_TRUE_REWARD = []
                list_reward = list(self.REWARDS_INFO_DICT.keys())
                list_reward.sort()
                for reward_name in list_reward:
                    if self.REWARDS_INFO_DICT[reward_name][0]:
                        self.LIST_TRUE_REWARD.append(reward_name)

    def __init__(self, path_set):

        self.path_set = path_set

        self.para_set = self.load_conf(os.path.join(self.path_set.PATH_TO_CONF, self.path_set.CITYFLOW_AGENT_CONF))

        self.eng = map_computor.reset(self.para_set)

        start_phase = map_computor.start_sumo(self.eng)

        self.global_dic_vehicles = collections.OrderedDict()
        self.global_dic_location_vehicles = collections.OrderedDict()
        self.global_dic_this_node_vehicles = collections.OrderedDict()
        self.global_dic_waiting_time_vehicles = collections.OrderedDict()
        self.global_list_node_lane_vehicle = collections.OrderedDict()

        for node_id_1 in map_computor.get_node_id_list():
            self.global_dic_location_vehicles[node_id_1] = collections.OrderedDict()
            self.global_dic_this_node_vehicles[node_id_1] = collections.OrderedDict()
            for lane in map_computor.global_listLanes[node_id_1]:
                self.global_dic_waiting_time_vehicles[lane] = collections.OrderedDict()
            for lane in map_computor.global_all_lanes_each_node[node_id_1]:
                self.global_list_node_lane_vehicle[lane] = []


        self.current_phase = collections.OrderedDict()
        self.current_phase_duration = collections.OrderedDict()
        for node_id_index in range(len(map_computor.get_node_id_list())):
            node_id = map_computor.get_node_id_list()[node_id_index]
            self.current_phase[node_id] = start_phase[node_id_index]
            self.current_phase_duration[node_id] = 0
        self.state = collections.OrderedDict()     # it's global!!!
        self.all_vehicles_location_enter_time_dict = collections.OrderedDict()
        self.all_vehicles_this_node_enter_time_dict = collections.OrderedDict()
        for node_id_1 in map_computor.get_node_id_list():
            self.all_vehicles_location_enter_time_dict[node_id_1] = collections.OrderedDict()
            self.all_vehicles_this_node_enter_time_dict[node_id_1] = collections.OrderedDict()

        self.update_state()
        self.update_vehicles()
        self.update_vehicles_location()
        self.update_vehicle_arrive_leave_time()

        self.f_log_rewards = os.path.join(self.path_set.PATH_TO_OUTPUT, "log_rewards.txt")
        self.f_log_rewards_control = os.path.join(self.path_set.PATH_TO_OUTPUT, "control_log_rewards.txt")
        if not os.path.exists(self.f_log_rewards):
            f = open(self.f_log_rewards, 'w')
            list_reward_keys = np.sort(self.para_set.LIST_TRUE_REWARD)
            head_str = "node_id, current_time, action, " + ', '.join(list_reward_keys) + '\n'
            f.write(head_str)
            f.close()

        if not os.path.exists(self.f_log_rewards_control):
            f = open(self.f_log_rewards_control, 'w')
            head_str = "node_id, current_time, local_travel_time, average_local_travel_time" + '\n'
            f.write(head_str)
            f.close()


    def end_sumo(self, episode, current_time, file_name_travel_time, episode_time):
        map_computor.end_sumo(self.eng, episode, current_time, file_name_travel_time, episode_time)

    def end_sumo_test(self, episode, current_time, file_name_travel_time, episode_time):
        map_computor.end_sumo_test(self.eng, episode, current_time, file_name_travel_time, episode_time)

    def load_conf(self, conf_file):
        dic_paras = json.load(open(conf_file, "r"))
        return self.ParaSet(dic_paras)

    def get_observation(self):
        return self.state

    def get_current_time(self):
        return map_computor.get_current_time(self.eng)

    def get_current_phase(self):
        return self.current_phase

    def take_action(self, joint_action, p_indicator, warm_up):
        rewards_detail_dict_list =collections.OrderedDict()
        for node_id in joint_action.keys():
            current_phase_number = self.get_current_phase()[node_id]
            if self.current_phase[node_id] <= 2:     # 0, 2: straight
                if (self.current_phase_duration[node_id] < self.para_set.MIN_PHASE_TIME):
                    joint_action[node_id] = 0
                elif self.current_phase_duration[node_id] >= self.para_set.MAX_PHASE_TIME_STRAIGHT:
                    joint_action[node_id] = 1
            else:                                       # 1, 3: left
                if (self.current_phase_duration[node_id] < self.para_set.MIN_PHASE_TIME):
                    joint_action[node_id] = 0
                elif self.current_phase_duration[node_id] >= self.para_set.MAX_PHASE_TIME_LEFT:
                    joint_action[node_id] = 1

            rewards_detail_dict_list[node_id] = []

        for i in range(self.para_set.MIN_ACTION_TIME):
            joint_action_in_second = collections.OrderedDict()
            for node_id in joint_action.keys():
                joint_action_in_second[node_id] = 0
                current_phase_number = self.get_current_phase()[node_id]
                if joint_action[node_id] == 1 and i == 0:
                    joint_action_in_second[node_id] = 1
            self.current_phase, self.current_phase_duration, self.global_dic_vehicles, self.global_dic_location_vehicles, self.global_dic_this_node_vehicles, self.all_vehicles_location_enter_time_dict, self.all_vehicles_this_node_enter_time_dict, self.global_dic_waiting_time_vehicles, self.global_list_node_lane_vehicle\
                = map_computor.run(eng=self.eng,
                                   joint_action=joint_action_in_second,
                                   current_phase=self.get_current_phase(),
                                   current_phase_duration=self.current_phase_duration,
                                   global_vehicle_dict=self.global_dic_vehicles,
                                   global_dic_location_vehicles=self.global_dic_location_vehicles,
                                   global_dic_this_node_vehicles=self.global_dic_this_node_vehicles,
                                   all_vehicles_location_enter_time_dict=self.all_vehicles_location_enter_time_dict,
                                   all_vehicles_this_node_enter_time_dict=self.all_vehicles_this_node_enter_time_dict,
                                   global_dic_waiting_time_vehicles=self.global_dic_waiting_time_vehicles,
                                   global_list_node_lane_vehicle=self.global_list_node_lane_vehicle,
                                   rewards_info_dict=self.para_set.REWARDS_INFO_DICT,
                                   true_reward=self.para_set.LIST_TRUE_REWARD,
                                   f_log_rewards=self.f_log_rewards,
                                   rewards_detail_dict_list=rewards_detail_dict_list,
                                   reward_indicator=p_indicator,
                                   warm_up=warm_up)  # run 1s SUMO

        #reward, reward_detail_dict = self.cal_reward(action)
        global_reward = self.cal_reward_from_list(rewards_detail_dict_list)    # each node: reward of step0 + ... + reward of step5

        #self.update_vehicles()
        self.update_state()

        return global_reward, joint_action

    def get_control_reward(self):
        rewards_detail_dict_list = collections.OrderedDict()
        rewards_detail_this_node_dict_list = collections.OrderedDict()
        for node_id_1 in map_computor.get_node_id_list():
            rewards_detail_dict_list[node_id_1] = []
            rewards_detail_this_node_dict_list[node_id_1] = []
        map_computor.run_control(eng=self.eng,
                                 rewards_info_dict=self.para_set.REWARDS_CONTROL_INFO_DICT,
                                 rewards_this_node_info_dict=self.para_set.REWARDS_CONTROL_AUX_INFO_DICT,
                                 f_log_rewards_control=self.f_log_rewards_control,
                                 rewards_detail_dict_list=rewards_detail_dict_list,
                                 rewards_detail_this_node_dict_list=rewards_detail_this_node_dict_list,
                                 all_vehicles_location_enter_time_dict=self.all_vehicles_location_enter_time_dict,
                                 all_vehicles_this_node_enter_time_dict=self.all_vehicles_this_node_enter_time_dict)
        self.clear_local_travel_time()

        global_reward_aux = self.cal_reward_from_list(rewards_detail_dict_list)
        global_reward_main = self.cal_reward_from_list(rewards_detail_this_node_dict_list)
        return global_reward_aux, global_reward_main


    def take_action_pre_train(self, phase_time_now):
        rewards_detail_dict_list = collections.OrderedDict()
        joint_action = collections.OrderedDict()
        for i in range(4):
            for j in range(4):
                node_id = "node%d%d" % (i, j)
                current_phase_number = self.get_current_phase()[node_id]
                if (self.current_phase_duration < phase_time_now[current_phase_number]):
                    joint_action[node_id] = 0
                else:
                    joint_action[node_id] = 1

                rewards_detail_dict_list[node_id] = []

        for i in range(self.para_set.MIN_ACTION_TIME):
            joint_action_in_second = collections.OrderedDict()
            for node_id in joint_action.keys():
                joint_action_in_second[node_id] = 0
                current_phase_number = self.get_current_phase()[node_id]
                if joint_action[node_id] == 1 and i == 0:
                    joint_action_in_second[node_id] = 1
            self.current_phase, self.current_phase_duration, self.global_dic_vehicles = map_computor.run(joint_action=joint_action_in_second,
                                                                               current_phase=self.get_current_phase(),
                                                                               current_phase_duration=self.current_phase_duration,
                                                                               global_vehicle_dict=self.global_dic_vehicles,
                                                                               rewards_info_dict=self.para_set.REWARDS_INFO_DICT,
                                                                               f_log_rewards=self.f_log_rewards,
                                                                               rewards_detail_dict_list=rewards_detail_dict_list)  # run 1s SUMO
        global_reward = self.cal_reward_from_list(rewards_detail_dict_list)

        #self.update_vehicles()
        self.update_state()

        return global_reward, joint_action

    def update_vehicles(self):
        self.global_dic_vehicles = map_computor.update_vehicles_state(self.eng, self.global_dic_vehicles)

    def update_vehicles_location(self):
        self.global_dic_location_vehicles, self.global_dic_this_node_vehicles, self.all_vehicles_location_enter_time_dict, self.all_vehicles_this_node_enter_time_dict, self.global_dic_waiting_time_vehicles \
            = map_computor.update_vehicles_location(self.eng, self.global_dic_location_vehicles, self.global_dic_this_node_vehicles, self.all_vehicles_location_enter_time_dict, self.all_vehicles_this_node_enter_time_dict, self.global_dic_waiting_time_vehicles)

    def update_vehicle_arrive_leave_time(self):
        self.global_list_node_lane_vehicle = map_computor.update_dic_lane_vehicle_arrive_leave_time(self.eng, self.global_list_node_lane_vehicle)
    def clear_local_travel_time(self):
        self.all_vehicles_location_enter_time_dict = collections.OrderedDict()
        self.all_vehicles_this_node_enter_time_dict = collections.OrderedDict()
        for node_id_1 in map_computor.get_node_id_list():
            self.all_vehicles_location_enter_time_dict[node_id_1] = collections.OrderedDict()
            self.all_vehicles_this_node_enter_time_dict[node_id_1] = collections.OrderedDict()

    def update_state(self):
        status_trackers = map_computor.status_calculator(self.eng, self.global_dic_vehicles)
        DIC_PHASE_MAP = {
            1: 2,
            2: 3,
            3: 4,
            4: 1
        }
        for node_id, status_tracker in status_trackers.items():
            self.state[node_id] = State(
                queue_length=None,
                num_of_vehicles=np.reshape(np.array(status_tracker[1]), newshape=(1,) + State.D_NUM_OF_VEHICLES),
                waiting_time=None,
                map_feature=None,
                cur_phase=np.eye(State.D_CUR_PHASE[0], dtype=np.int16)[[self.current_phase[node_id]]], # one_hot
                next_phase=np.eye(State.D_NEXT_PHASE[0], dtype=np.int16)[[DIC_PHASE_MAP[self.current_phase[node_id]]]], # one_hot
                time_this_phase=None,
                if_terminal=None
            )
            '''
            self.state[node_id] = State(
                queue_length=np.reshape(np.array(status_tracker[0]), newshape=(1,) + State.D_QUEUE_LENGTH),
                num_of_vehicles=np.reshape(np.array(status_tracker[1]), newshape=(1,) + State.D_NUM_OF_VEHICLES),
                waiting_time=np.reshape(np.array(status_tracker[2]), newshape=(1,) + State.D_WAITING_TIME),
                map_feature=None,
                cur_phase=np.eye(State.D_CUR_PHASE[0], dtype=np.int16)[[self.current_phase[node_id]]], # one_hot
                next_phase=np.eye(State.D_NEXT_PHASE[0], dtype=np.int16)[[(self.current_phase[node_id] + 1) % (len(map_computor.get_node_phases(node_id)))]], # one_hot
                time_this_phase=np.reshape(np.array([self.current_phase_duration[node_id]]), newshape=(1,) + State.D_TIME_THIS_PHASE),
                if_terminal=False
            )
            '''

    # it looks useless
    def cal_reward(self, action):
        # get directly from sumo
        reward, reward_detail_dict = map_computor.get_rewards_from_sumo(self.global_dic_vehicles, action, self.para_set.REWARDS_INFO_DICT)
        return reward*(1-0.8), reward_detail_dict

    def cal_reward_from_list(self, global_reward_detail_dict_list):
        global_reward = collections.OrderedDict()
        for node_id, reward_detail_dict_list in global_reward_detail_dict_list.items():
            reward = map_computor.get_rewards_from_dict_list(reward_detail_dict_list)
            global_reward[node_id] = reward*(1-0.8)
        return global_reward


if __name__ == '__main__':
    pass
