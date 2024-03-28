# -*- coding: utf-8 -*-

'''
Controling agent, mainly choosing actions

'''

import json
import os
import shutil
import map_computor


class State(object):
    # ==========================

    D_QUEUE_LENGTH = (12,)
    D_NUM_OF_VEHICLES = (12,)
    D_WAITING_TIME = (12,)
    D_MAP_FEATURE = (150,150,8,)
    D_CUR_PHASE = (5,)
    D_NEXT_PHASE = (5,)
    D_TIME_THIS_PHASE = (1,)
    D_IF_TERMINAL = (1,)
    D_HISTORICAL_TRAFFIC = (6,)
    

    # ==========================

    def __init__(self,
                 queue_length, num_of_vehicles, waiting_time, map_feature,
                 cur_phase,
                 next_phase,
                 time_this_phase,
                 if_terminal):

        self.queue_length = queue_length
        self.num_of_vehicles = num_of_vehicles
        self.waiting_time = waiting_time
        self.map_feature = map_feature

        self.cur_phase = cur_phase
        self.next_phase = next_phase
        self.time_this_phase = time_this_phase

        self.if_terminal = if_terminal

        self.historical_traffic = None


class Agent(object):

    class ParaSet:

        def __init__(self, dic_paras):
            for key, value in dic_paras.items():
                setattr(self, key, value)

            if hasattr(self, "STATE_FEATURE"):
                self.LIST_STATE_FEATURE = []
                list_state_feature_names = list(self.STATE_FEATURE.keys())
                list_state_feature_names.sort()
                for feature_name in list_state_feature_names:
                    if self.STATE_FEATURE[feature_name]:
                        self.LIST_STATE_FEATURE.append(feature_name)

    def __init__(self, num_phases,
                 path_set, node_id):

        self.path_set = path_set
        self.para_set = self.load_conf(os.path.join(self.path_set.PATH_TO_CONF, self.path_set.AGENT_CONF))

        self.node_id = node_id
        self.num_phases = num_phases
        self.control_state = []
        self.state = None
        self.control_action = None
        self.action = None
        self.control_memory = []
        self.memory = [[], [], []]
        self.average_reward = None
        if self.node_id == '11111111111':
            self.neighbor = [-1, -1, -1, -1, -1]
        else:
            self.neighbor = map_computor.find_neighbors()[self.node_id]    # [self, other, other, other, other]

        self.feature_name_input = []
        for neighbor in range(len(self.neighbor)):
            for feature_name in self.para_set.LIST_STATE_FEATURE:
                self.feature_name_input.append(feature_name + str(neighbor))

    def load_conf(self, conf_file):

        dic_paras = json.load(open(conf_file, "r"))
        return self.ParaSet(dic_paras)

    def get_control_state(self, global_state):

        ''' set state for agent '''
        self.control_state = []
        node_id_list = map_computor.get_node_id_list()
        for m in self.neighbor:
            if m != -1:
                self.control_state.append(global_state[node_id_list[m]])
            else:
                self.control_state.append(-1)
        # self.control_state = [global_state[m] for m in self.neighbor]
        return self.control_state

    def get_warm_up_control_state(self, state):

        ''' set state for agent '''
        self.control_state = state
        return self.control_state

    def get_state(self, state, count):

        ''' set state for agent '''
        self.state = state
        return state

    def get_next_control_state(self, state, count):

        return self.control_state

    def get_next_state(self, state, count):

        return state

    def choose(self, count, if_pretrain, p_index):

        ''' choose the best action for current state '''

        pass

    def remember(self, state, action, reward, next_state, p_index):
        ''' log the history separately '''

        pass

    def clear_memory(self):

        pass

    def reset_update_count(self):

        pass

    def update_network(self, net_index, warm_up, if_pretrain, use_average, current_time):
        pass

    def update_network_bar(self, net_index, warm_up):
        pass

    def forget(self):
        pass

    def batch_predict(self,file_name="temp"):
        pass
