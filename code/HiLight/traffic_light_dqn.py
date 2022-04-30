# -*- coding: utf-8 -*-

'''

python TrafficLightDQN.py SEED setting_memo

SEED: random number for initializing the experiment
setting_memo: the folder name for this experiment
    The conf, data files will should be placed in conf/setting_memo, data/setting_memo respectively
    The records, model files will be generated in records/setting_memo, model/setting_memo respectively

'''


import copy
import json
import shutil

import os
import time
import math
import map_computor as map_computor
from deeplight_agent import DeeplightAgent

from cityflow_agent import CityFlowAgent
import xml.etree.ElementTree as ET
import numpy as np
import random
import copy
import collections
from keras.utils import np_utils,to_categorical

TRAFFIC_FILE = "traffic_file.json"
PATH_TO_CONF = "conf/...folder of roadnet"

class TrafficLightDQN:

    DIC_AGENTS = {
        "Deeplight": DeeplightAgent,
    }

    NO_PRETRAIN_AGENTS = []

    class ParaSet:

        def __init__(self, dic_paras):
            for key, value in dic_paras.items():
                setattr(self, key, value)

    class PathSet:

        # ======================================= conf files ========================================
        EXP_CONF = "exp.conf"
        CITYFLOW_AGENT_CONF = "cityflow_agent.conf"
        PATH_TO_CFG_TMP = os.path.join("data", "tmp")
        # ======================================= conf files ========================================

        def __init__(self, path_to_conf, path_to_data, path_to_output, path_to_model):

            self.PATH_TO_CONF = path_to_conf
            self.PATH_TO_DATA = path_to_data
            self.PATH_TO_OUTPUT = path_to_output
            self.PATH_TO_MODEL = path_to_model

            if not os.path.exists(self.PATH_TO_OUTPUT):
                os.makedirs(self.PATH_TO_OUTPUT)
            if not os.path.exists(self.PATH_TO_MODEL):
                os.makedirs(self.PATH_TO_MODEL)

            dic_paras = json.load(open(os.path.join(self.PATH_TO_CONF, self.EXP_CONF), "r"))
            self.AGENT_CONF = "{0}_agent.conf".format(dic_paras["MODEL_NAME"].lower())
            self.TRAFFIC_FILE = dic_paras["TRAFFIC_FILE"]
            self.TRAFFIC_FILE_PRETRAIN = dic_paras["TRAFFIC_FILE_PRETRAIN"]

    def __init__(self, memo, f_prefix, has_trained, load_model_name):

        self.path_set = self.PathSet(os.path.join("conf", memo),
                                     os.path.join("data", memo),
                                     os.path.join("records", memo, f_prefix),
                                     os.path.join("model", memo, f_prefix))

        self.para_set = self.load_conf(conf_file=os.path.join(self.path_set.PATH_TO_CONF, self.path_set.EXP_CONF))

        self.update_index = 0

        self.untrained_agent = collections.OrderedDict()
        self.trained_agent = collections.OrderedDict()

        if not has_trained:
            self.warm_up_agent = self.DIC_AGENTS[self.para_set.MODEL_NAME](num_phases=4,
                                                                        num_actions=2,
                                                                        control_num_actions=self.para_set.SUB_POLICY_NUMBER,
                                                                        path_set=self.path_set,
                                                                        node_id='11111111111',
                                                                        warm_up=True)


            for node_id in map_computor.get_node_id_list():
                self.untrained_agent[node_id] = self.DIC_AGENTS[self.para_set.MODEL_NAME](num_phases=4,
                                                                                        num_actions=2,
                                                                                        control_num_actions=self.para_set.SUB_POLICY_NUMBER,
                                                                                        path_set=self.path_set,
                                                                                        node_id=node_id,
                                                                                        warm_up=False)

        else:
            self.warm_up_agent = self.DIC_AGENTS[self.para_set.MODEL_NAME](num_phases=4,
                                                                            num_actions=2,
                                                                            control_num_actions=self.para_set.SUB_POLICY_NUMBER,
                                                                            path_set=self.path_set,
                                                                            node_id='11111111111',
                                                                            warm_up=False,
                                                                            load_model_from=load_model_name)


            for node_id in map_computor.get_node_id_list():
                self.trained_agent[node_id] = self.DIC_AGENTS[self.para_set.MODEL_NAME](num_phases=4,
                                                                                        num_actions=2,
                                                                                        control_num_actions=self.para_set.SUB_POLICY_NUMBER,
                                                                                        path_set=self.path_set,
                                                                                        node_id=node_id,
                                                                                        warm_up=False)


    def load_conf(self, conf_file):

        dic_paras = json.load(open(conf_file, "r"))
        return self.ParaSet(dic_paras)

    def check_if_need_pretrain(self):

        if self.para_set.MODEL_NAME in self.NO_PRETRAIN_AGENTS:
            return False
        else:
            return True

    def _generate_pre_train_ratios(self, phase_min_time, em_phase):
        phase_traffic_ratios = [phase_min_time]

        # generate how many varients for each phase
        for i, phase_time in enumerate(phase_min_time):
            if i == em_phase:
                for j in range(1, 5, 1):
                    gen_phase_time = copy.deepcopy(phase_min_time)
                    gen_phase_time[i] += j
                    phase_traffic_ratios.append(gen_phase_time)
            else:
                # pass
                for j in range(1, 5, 1):
                    gen_phase_time = copy.deepcopy(phase_min_time)
                    gen_phase_time[i] += j
                    phase_traffic_ratios.append(gen_phase_time)
            for j in range(5, 20, 5):
                gen_phase_time = copy.deepcopy(phase_min_time)
                gen_phase_time[i] += j
                phase_traffic_ratios.append(gen_phase_time)

        return phase_traffic_ratios

    @staticmethod
    def _set_traffic_file(sumo_config_file_tmp_name, sumo_config_file_output_name, list_traffic_file_name):

        # update sumocfg
        sumo_cfg = ET.parse(sumo_config_file_tmp_name)
        config_node = sumo_cfg.getroot()
        input_node = config_node.find("input")
        for route_files in input_node.findall("route-files"):
            input_node.remove(route_files)
        input_node.append(
            ET.Element("route-files", attrib={"value": ",".join(list_traffic_file_name)}))
        sumo_cfg.write(sumo_config_file_output_name)


    def copy_networks(self):
        for node_id in map_computor.get_node_id_list():
            self.untrained_agent[node_id].q_network = self.warm_up_agent.q_network

    def discount_rewards(self, rewards, gamma):
        running_total = 0
        discounted = np.zeros_like(rewards)
        for r in reversed(range(len(rewards))):
            running_total = running_total * gamma + rewards[r]
            discounted[r] = running_total
        return discounted


    def train(self, if_pretrain, use_average, episode):
        dic_deeplight = json.load(open(os.path.join(PATH_TO_CONF, "deeplight_agent.conf"), "r"))
        self.warm_up_agent.para_set.EPSILON = dic_deeplight["EPSILON"]
        json.dump(dic_deeplight, open(os.path.join(PATH_TO_CONF, "deeplight_agent.conf"), "w"), indent=4)

        total_run_cnt = self.para_set.RUN_COUNTS

        control_period = self.para_set.UPDATE_CONTROL_CHOICE

        # initialize output streams
        file_name_memory = os.path.join(self.path_set.PATH_TO_OUTPUT, "memories.txt")
        file_name_memory_control = os.path.join(self.path_set.PATH_TO_OUTPUT, "control_memories.txt")

        # start sumo
        s_agent = CityFlowAgent(self.path_set)
        current_time = s_agent.get_current_time()  # in seconds
        control_update_index = 0

        meta_z = collections.OrderedDict()
        meta_rewards_main = collections.OrderedDict()
        meta_rewards_aux = collections.OrderedDict()
        meta_states = collections.OrderedDict()
        meta_states_aux = collections.OrderedDict()
        duoyu_states = collections.OrderedDict()
        duoyu_states_aux = collections.OrderedDict()
        duoyu_z = collections.OrderedDict()

        for node_id_1 in map_computor.get_node_id_list():
            meta_z[node_id_1] = []
            meta_rewards_main[node_id_1] = []
            meta_rewards_aux[node_id_1] = []
            meta_states[node_id_1] = []
            duoyu_states[node_id_1] = []
            meta_states_aux[node_id_1] = []
            duoyu_states_aux[node_id_1] = []
            duoyu_z[node_id_1] = []

        # start experiment
        while current_time < total_run_cnt:

            f_memory = open(file_name_memory, "a")
            f_memory_control = open(file_name_memory_control, "a")

            # get state
            global_state = s_agent.get_observation()
            global_state_copy = copy.deepcopy(global_state)
            global_action_pred = collections.OrderedDict()
            global_q_values = collections.OrderedDict()

            if (current_time == 1) | (current_time - self.update_index > control_period):
                if current_time != 1:
                    self.update_index = current_time
                global_action_control = collections.OrderedDict()
                global_control_state = global_state_copy
                for node_id_1, state in global_state_copy.items():
                    state = self.untrained_agent[node_id_1].get_state(state, current_time)
                    state_aux = self.untrained_agent[node_id_1].get_control_state(global_state_copy)
                    state_11 = self.warm_up_agent.get_state(state, current_time)
                    feature = self.warm_up_agent.convert_state_to_feature(state_11)
                    p_z = self.warm_up_agent.control_Pi.get_dist(feature)[0]
                    z = np.random.choice(range(3), p=p_z)
                    global_action_control[node_id_1] = z
                    meta_z[node_id_1].append(to_categorical(z, 3))
                    meta_states[node_id_1].append(list(feature[0]))

                    feature_aux = self.warm_up_agent.convert_control_state_to_feature(state_aux)
                    meta_states_aux[node_id_1].append(list(feature_aux[0]))

            for node_id_1, state in global_state_copy.items():
                state_1 = self.untrained_agent[node_id_1].get_state(state, current_time)
                warmup_state = self.warm_up_agent.get_state(state, current_time)
                action_taken = global_action_control[node_id_1]

                action_pred, q_values = self.warm_up_agent.choose(count=current_time, if_pretrain=if_pretrain, p_index=action_taken)

                global_action_pred[node_id_1] = action_pred
                global_q_values[node_id_1] = q_values

            global_reward, global_action, _ = s_agent.take_action(global_action_pred, global_action_control, False)

            # get next state
            global_next_state = s_agent.get_observation()
            for node_id_1, next_state in global_next_state.items():
                next_state = self.untrained_agent[node_id_1].get_next_state(next_state, current_time)

                # remember
                state = global_state_copy[node_id_1]
                reward = global_reward[node_id_1]
                action = global_action[node_id_1]
                action_taken = global_action_control[node_id_1]
                self.warm_up_agent.remember_by_index(state, action, reward, next_state, action_taken)

                q_values = global_q_values[node_id_1]
                # output to std out and file
                memory_str = 'node_id=%s\ttime = %d\tchosen_policy = %d\taction = %d\tcurrent_phase = %d\tnext_phase = %d\treward = %f' \
                             '\t%s' \
                             % (node_id_1, current_time, global_action_control[node_id_1], action,
                                np.nonzero(state.cur_phase[0])[0][0],
                                np.nonzero(next_state.cur_phase[0])[0][0],
                                reward, repr(q_values))
                print(memory_str)
                f_memory.write(memory_str + "\n")
            f_memory.close()
            current_time = s_agent.get_current_time()

            if current_time - self.update_index > control_period:
                global_reward_control_aux, global_reward_control_main = s_agent.get_control_reward()
                for node_id_1 in map_computor.get_node_id_list():
                    meta_rewards_aux[node_id_1].append(global_reward_control_aux[node_id_1])
                    meta_rewards_main[node_id_1].append(global_reward_control_main[node_id_1])

                if not if_pretrain:
                # update network
                    for net_index in range(self.para_set.SUB_POLICY_NUMBER):
                        self.warm_up_agent.update_network(net_index, False, if_pretrain, use_average, current_time)
                        self.warm_up_agent.update_network_bar(net_index)
                    print(self.warm_up_agent.para_set.EPSILON)


                global_next_control_state = global_next_state
                for node_id_1, next_control_state in global_next_control_state.items():
                    # remember
                    state_control = global_control_state[node_id_1]
                    reward_control_main = global_reward_control_main[node_id_1]
                    reward_control_aux = global_reward_control_aux[node_id_1]
                    action_control = global_action_control[node_id_1]

                    control_memory_str = 'choose sub_policy!!!\nnode_id=%s\ttime = %d\tchoose_subpolicy_index = %d\treward_main = %f\treward_aux = %f' \
                                % (node_id_1, current_time, action_control, reward_control_main, reward_control_aux)
                    print(control_memory_str)
                    f_memory_control.write(control_memory_str + "\n")
                f_memory_control.close()

            if current_time - control_update_index > 500:
                control_update_index = current_time
                for node_id_1 in map_computor.get_node_id_list():
                    if len(meta_rewards_main[node_id_1]) == 0:
                        continue
                    if len(meta_rewards_main[node_id_1]) != len(meta_states[node_id_1]):
                        lens = len(meta_rewards_main[node_id_1])
                        duoyu_states[node_id_1].append(meta_states[node_id_1][-1])
                        duoyu_states_aux[node_id_1].append(meta_states_aux[node_id_1][-1])
                        duoyu_z[node_id_1].append(meta_z[node_id_1][-1])
                        meta_states[node_id_1] = meta_states[node_id_1][0:lens]
                        meta_states_aux[node_id_1] = meta_states_aux[node_id_1][0:lens]
                        meta_z[node_id_1] = meta_z[node_id_1][0:lens]

                    meta_z[node_id_1] = np.array(meta_z[node_id_1])
                    meta_rewards_main[node_id_1] = np.array(meta_rewards_main[node_id_1])
                    meta_rewards_aux[node_id_1] = np.array(meta_rewards_aux[node_id_1])
                    meta_states[node_id_1] = np.array(meta_states[node_id_1])
                    meta_states_aux[node_id_1] = np.array(meta_states_aux[node_id_1])
                    gamma = self.warm_up_agent.para_set.GAMMA
                    targets_main = self.discount_rewards(meta_rewards_main[node_id_1], gamma)
                    targets_aux = self.discount_rewards(meta_rewards_aux[node_id_1], gamma)
                    self.warm_up_agent.control_V_main.update(meta_states[node_id_1], targets_main)
                    self.warm_up_agent.control_V_aux.update(meta_states_aux[node_id_1], targets_aux)
                    meta_advantages_main = meta_rewards_main[node_id_1] - self.warm_up_agent.control_V_main.get(meta_states[node_id_1])
                    meta_advantages_aux = meta_rewards_aux[node_id_1] - self.warm_up_agent.control_V_aux.get(meta_states_aux[node_id_1])
                    gradient_main = self.warm_up_agent.control_Pi.get_gradient(meta_states[node_id_1],
                                                                               meta_z[node_id_1],
                                                                               meta_advantages_main)
                    gradient_aux = self.warm_up_agent.control_Pi.get_gradient(meta_states[node_id_1], meta_z[node_id_1],
                                                                              meta_advantages_aux)
                    weight, dot_sum = self.warm_up_agent.control_Pi.update_weight(gradient_main, gradient_aux, node_id_1)
                    meta_advantages = meta_advantages_main + weight * meta_advantages_aux
                    self.warm_up_agent.control_Pi.update(meta_states[node_id_1], meta_z[node_id_1], meta_advantages)


                meta_states = copy.deepcopy(duoyu_states)
                meta_states_aux = copy.deepcopy(duoyu_states_aux)
                meta_z = copy.deepcopy(duoyu_z)
                for node_id_1 in map_computor.get_node_id_list():
                    meta_rewards_main[node_id_1] = []
                    meta_rewards_aux[node_id_1] = []
                    duoyu_states[node_id_1] = []
                    duoyu_states_aux[node_id_1] = []
                    duoyu_z[node_id_1] = []

        for node_id_1 in map_computor.get_node_id_list():
            if len(meta_rewards_main[node_id_1]) == 0:
                continue
            if len(meta_rewards_main[node_id_1]) != len(meta_states[node_id_1]):
                lens = len(meta_rewards_main[node_id_1])
                duoyu_states[node_id_1].append(meta_states[node_id_1][-1])
                duoyu_states_aux[node_id_1].append(meta_states_aux[node_id_1][-1])
                duoyu_z[node_id_1].append(meta_z[node_id_1][-1])
                meta_states[node_id_1] = meta_states[node_id_1][0:lens]
                meta_states_aux[node_id_1] = meta_states_aux[node_id_1][0:lens]
                meta_z[node_id_1] = meta_z[node_id_1][0:lens]

            meta_z[node_id_1] = np.array(meta_z[node_id_1])
            meta_rewards_main[node_id_1] = np.array(meta_rewards_main[node_id_1])
            meta_rewards_aux[node_id_1] = np.array(meta_rewards_aux[node_id_1])
            meta_states[node_id_1] = np.array(meta_states[node_id_1])
            meta_states_aux[node_id_1] = np.array(meta_states_aux[node_id_1])
            gamma = self.warm_up_agent.para_set.GAMMA
            targets_main = self.discount_rewards(meta_rewards_main[node_id_1], gamma)
            targets_aux = self.discount_rewards(meta_rewards_aux[node_id_1], gamma)
            self.warm_up_agent.control_V_main.update(meta_states[node_id_1], targets_main)
            self.warm_up_agent.control_V_aux.update(meta_states_aux[node_id_1], targets_aux)
            meta_advantages_main = meta_rewards_main[node_id_1] - self.warm_up_agent.control_V_main.get(
                meta_states[node_id_1])
            meta_advantages_aux = meta_rewards_aux[node_id_1] - self.warm_up_agent.control_V_aux.get(
                meta_states_aux[node_id_1])
            gradient_main = self.warm_up_agent.control_Pi.get_gradient(meta_states[node_id_1], meta_z[node_id_1],
                                                                 meta_advantages_main)
            gradient_aux = self.warm_up_agent.control_Pi.get_gradient(meta_states[node_id_1], meta_z[node_id_1],
                                                                meta_advantages_aux)
            weight, dot_sum = self.warm_up_agent.control_Pi.update_weight(gradient_main, gradient_aux, node_id_1)
            meta_advantages = meta_advantages_main + weight * meta_advantages_aux
            self.warm_up_agent.control_Pi.update(meta_states[node_id_1], meta_z[node_id_1], meta_advantages)


        for net_index in range(self.para_set.SUB_POLICY_NUMBER):
            self.warm_up_agent.save_model(episode, net_index)

        for _, agent in self.untrained_agent.items():
            agent.reset_q_bar_outdated()
        self.update_index = 0

        s_agent.end_sumo(episode, current_time)
        dic_deeplight = json.load(open(os.path.join(PATH_TO_CONF, "deeplight_agent.conf"), "r"))
        print(dic_deeplight["EPSILON"])
        json.dump(dic_deeplight, open(os.path.join(PATH_TO_CONF, "deeplight_agent.conf"), "w"), indent=4)
        print("############################# episode: %s finished." % episode)
        print("END")

        self.test_test(episode)

    def test_test(self, episode):
        self.warm_up_agent.para_set.EPSILON = 0

        total_run_cnt = self.para_set.RUN_COUNTS

        control_period = self.para_set.UPDATE_CONTROL_CHOICE

        file_name_memory = os.path.join(self.path_set.PATH_TO_OUTPUT, "test_memories.txt")
        file_name_memory_control = os.path.join(self.path_set.PATH_TO_OUTPUT, "test_control_memories.txt")

        # start sumo
        s_agent = CityFlowAgent(self.path_set)
        current_time = s_agent.get_current_time()
        x = []
        y = []

        # start experiment
        while current_time < total_run_cnt:

            f_memory = open(file_name_memory, "a")
            f_memory_control = open(file_name_memory_control, "a")

            # get state
            global_state = s_agent.get_observation()
            global_state_copy = copy.deepcopy(global_state)
            global_action_pred = collections.OrderedDict()
            global_q_values = collections.OrderedDict()

            if (current_time == 1) | (current_time - self.update_index > control_period):

                global_action_control = collections.OrderedDict()
                global_q_values_control = collections.OrderedDict()
                global_control_state = global_state_copy
                for node_id_1, state in global_state_copy.items():
                    state = self.untrained_agent[node_id_1].get_state(state, current_time)
                    state_11 = self.warm_up_agent.get_state(state, current_time)
                    global_control_state[node_id_1] = state
                    feature = self.warm_up_agent.convert_state_to_feature(state_11)
                    p_z = self.warm_up_agent.control_Pi.get_dist(feature)[0]
                    z = list(p_z).index(max(list(p_z)))
                    global_action_control[node_id_1] = z

            for node_id_1, state in global_state_copy.items():
                state_1 = self.untrained_agent[node_id_1].get_state(state, current_time)
                warmup_state = self.warm_up_agent.get_state(state, current_time)
                action_taken = global_action_control[node_id_1]

                action_pred, q_values = self.warm_up_agent.choose(count=current_time, if_pretrain=False, p_index=action_taken)

                global_action_pred[node_id_1] = action_pred
                global_q_values[node_id_1] = q_values

            global_reward, global_action, vehicle_number = s_agent.take_action(global_action_pred,
                                                                               global_action_control, False)

            # get next state
            global_next_state = s_agent.get_observation()
            for node_id_1, next_state in global_next_state.items():
                next_state = self.untrained_agent[node_id_1].get_next_state(next_state, current_time)

                # remember
                state = global_state_copy[node_id_1]
                reward = global_reward[node_id_1]
                action = global_action[node_id_1]
                action_taken = global_action_control[node_id_1]
                q_values = global_q_values[node_id_1]
                # output to std out and file
                memory_str = 'node_id=%s\ttime = %d\tchosen_policy = %d\taction = %d\tcurrent_phase = %d\tnext_phase = %d\treward = %f' \
                             '\t%s' \
                             % (node_id_1, current_time, global_action_control[node_id_1], action,
                                np.nonzero(state.cur_phase[0])[0][0],
                                np.nonzero(next_state.cur_phase[0])[0][0],
                                reward, repr(q_values))
                print(memory_str)
                f_memory.write(memory_str + "\n")
            f_memory.close()

            current_time = s_agent.get_current_time()  # in seconds


            if current_time - self.update_index > control_period:
                self.update_index = current_time

                global_next_control_state = global_next_state
                global_reward_control_aux, global_reward_control_main = s_agent.get_control_reward()
                for node_id_1, next_control_state in global_next_control_state.items():
                    next_control_state = self.untrained_agent[node_id_1].get_next_state(next_control_state, current_time)

                    # remember
                    state_control = global_control_state[node_id_1]
                    reward_control_main = global_reward_control_main[node_id_1]
                    reward_control_aux = global_reward_control_aux[node_id_1]
                    action_control = global_action_control[node_id_1]
                    # output to std out and file
                    control_memory_str = 'choose sub_policy!!!\nnode_id=%s\ttime = %d\tchoose_subpolicy_index = %d\treward_main = %f\treward_aux = %f' \
                                         % (node_id_1, current_time, action_control, reward_control_main, reward_control_aux)
                    print(control_memory_str)
                    f_memory_control.write(control_memory_str + "\n")
                f_memory_control.close()


        self.update_index = 0
        s_agent.end_sumo_test(episode, current_time)
        print("############################# episode: %s finished." % episode)
        print("END")

    def clear_memory(self):
        self.warm_up_agent.clear_memory()


def main(memo, f_prefix):

    player = TrafficLightDQN(memo, f_prefix, has_trained=False, load_model_name=[])

    player.clear_memory()

    dic_deeplight = json.load(open(os.path.join(PATH_TO_CONF, "deeplight_agent.conf"), "r"))
    dic_deeplight["EPSILON"] = 0.4
    json.dump(dic_deeplight, open(os.path.join(PATH_TO_CONF, "deeplight_agent.conf"), "w"), indent=4)

    for i in range(200):
        episode = str(i)
        player.train(if_pretrain=False, use_average=False, episode=episode)


def test(memo, f_prefix, sumo_cmd_str):
    for i in range(200):
        load_model_name = []
        for j in range(3):
            model_name = "{0}_{1}.h5".format(str(i), str(j))
            load_model_name.append(model_name)
        control_model_name = "{0}_control.h5".format(str(i))
        load_model_name.append(control_model_name)
        player = TrafficLightDQN(memo, f_prefix, has_trained=True, load_model_name=load_model_name)
        player.test(sumo_cmd_str, str(i))

