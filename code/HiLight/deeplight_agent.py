# -*- coding: utf-8 -*-

'''

Deep reinforcement learning agent

'''

import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Multiply, Add, Lambda, Reshape, Subtract
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers.merge import concatenate, add
from keras import backend as K
import random
import os
import collections
import time

from network_agent import NetworkAgent, conv2d_bn, Selector, State
import map_computor
import tensorflow as tf
import math

MEMO = "Deeplight"


class DeeplightAgent(NetworkAgent):

    def __init__(self,
                 num_phases,
                 num_actions,
                 control_num_actions,
                 path_set,
                 node_id,
                 warm_up,
                 load_model_from=[]):

        super(DeeplightAgent, self).__init__(
            num_phases=num_phases,
            path_set=path_set,
            node_id=node_id)

        self.num_actions = num_actions
        self.control_num_actions = control_num_actions
        self.node_id = node_id
        self.warm_up = warm_up

        # --- my modification ---
        if warm_up:
            self.q_network = []
            self.q_network_bar = []
            for i in range(control_num_actions):
                self.q_network.append(self.build_network(num_neighbor=1, num_action=self.num_actions))
                self.q_network_bar.append(self.build_network(num_neighbor=1, num_action=self.num_actions))
            self.control_Pi = PPOPolicyNetwork(num_features=22, num_actions=self.control_num_actions, layer_size=32,
                                               epsilon=0.2, learning_rate=0.0003)
            self.control_V_main = ValueNetwork(num_features=22, hidden_size=32, learning_rate=0.001)
            self.control_V_aux = ValueNetwork(num_features=110, hidden_size=32, learning_rate=0.001)


        if len(load_model_from) == 4:
            self.q_network = []
            path_to_model = "/home/lab/xby/reward_17280/model/three/['xby_v2.trips.xml']_11_23_13_38_18"

            for i in range(3):
                self.load_model(path_to_model, load_model_from[i], layer_dict={'conv2d_bn':conv2d_bn, 'Selector':Selector}, network_index=i)
            self.load_control_model(path_to_model, load_model_from[3], layer_dict={'conv2d_bn':conv2d_bn, 'Selector':Selector})



        self.update_outdated = [0, 0, 0]

        self.q_bar_outdated = [0, 0, 0]
        self.control_q_bar_outdated = 0


        if not self.para_set.SEPARATE_MEMORY:
                self.control_memory = self.control_build_memory()
                self.memory = self.build_memory()
        else:
                self.control_memory = self.build_memory_separate()
                self.memory = self.build_memory_separate()
        self.average_reward = None

    def reset_update_count(self):

        self.update_outdated = [0, 0, 0]
        self.q_bar_outdated = [0, 0, 0]

    def reset_q_bar_outdated(self):
        self.q_bar_outdated = [0, 0, 0]

    def reset_control_q_bar_outdated(self):
        self.control_q_bar_outdated = 0

    def set_update_outdated(self):

        self.update_outdated = - 2*self.para_set.UPDATE_PERIOD
        self.q_bar_outdated = 2*self.para_set.UPDATE_Q_BAR_FREQ

    def convert_state_to_input(self, state):

        ''' convert a state struct to the format for neural network input'''
        return [getattr(state, feature_name)
                    for feature_name in self.para_set.LIST_STATE_FEATURE]

    def convert_state_to_feature(self, state):

        ''' convert a state struct to the format for neural network input'''
        feature = [getattr(state, feature_name) for feature_name in self.para_set.LIST_STATE_FEATURE]
        return np.concatenate(feature, axis=1)

    def convert_control_state_to_input(self, control_state):

        ''' convert a state struct to the format for neural network input'''
        converted_control_state = []

        for state_index in range(len(control_state)):
            if control_state[state_index] != -1:
                for feature_name in self.para_set.LIST_STATE_FEATURE:
                    converted_control_state.append(getattr(control_state[state_index], feature_name))
            else:
                converted_control_state.append(np.array([[0, 0, 0, 0, 0]], dtype='int16'))
                converted_control_state.append(np.array([[0, 0, 0, 0, 0]], dtype='int16'))
                converted_control_state.append(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='int64'))

        return converted_control_state

    def convert_control_state_to_feature(self, control_state):

        ''' convert a state struct to the format for neural network input'''
        converted_control_state = []

        for state_index in range(len(control_state)):
            if control_state[state_index] != -1:
                for feature_name in self.para_set.LIST_STATE_FEATURE:
                    converted_control_state.append(getattr(control_state[state_index], feature_name))
            else:
                converted_control_state.append(np.array([[0, 0, 0, 0, 0]], dtype='int16'))
                converted_control_state.append(np.array([[0, 0, 0, 0, 0]], dtype='int16'))
                converted_control_state.append(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='int64'))
        return np.concatenate(converted_control_state, axis=1)


    def build_network(self, num_neighbor, num_action):

        '''Initialize a Q network'''

        # initialize feature node
        # dic_input_node = collections.OrderedDict()
        dic_input_node = collections.OrderedDict()
        for neighbor in range(num_neighbor):
            for feature_name in self.para_set.LIST_STATE_FEATURE:
                feature_name_input = feature_name + str(neighbor)
                dic_input_node[feature_name_input] = Input(shape=getattr(State, "D_"+feature_name.upper()),
                                                        name="input_neighbor_"+str(neighbor)+feature_name)

        # add cnn to image features
        dic_flatten_node = collections.OrderedDict()
        for k, _ in dic_input_node.items():
            dic_flatten_node[k] = dic_input_node[k]

        # concatenate features
        list_all_flatten_feature = []
        for feature_name in dic_flatten_node.keys():
            list_all_flatten_feature.append(dic_flatten_node[feature_name])
        if len(list_all_flatten_feature) > 1:
            all_flatten_feature = concatenate(list_all_flatten_feature, axis=1, name="all_flatten_feature")
        else:
            all_flatten_feature = list_all_flatten_feature[0]

        # shared dense layer
        shared_dense = self._shared_network_structure(all_flatten_feature, self.para_set.D_DENSE)       # D_DENSE = 12

        # build phase selector layer
        if "cur_phase" in self.para_set.LIST_STATE_FEATURE and self.para_set.PHASE_SELECTOR:
            list_selected_q_values = []
            for phase in range(self.num_phases):
                locals()["q_values_{0}".format(phase)] = self._separate_network_structure(
                    shared_dense, self.para_set.D_DENSE, self.num_actions, memo=phase)
                locals()["selector_{0}".format(phase)] = Selector(
                    phase, name="selector_{0}".format(phase))(dic_input_node["cur_phase"])
                locals()["q_values_{0}_selected".format(phase)] = Multiply(name="multiply_{0}".format(phase))(
                    [locals()["q_values_{0}".format(phase)],
                     locals()["selector_{0}".format(phase)]]
                )
                list_selected_q_values.append(locals()["q_values_{0}_selected".format(phase)])
            q_values = Add()(list_selected_q_values)
        else:
            q_values_state = self._separate_state_network_structure(shared_dense,
                self.para_set.D_DENSE, memo="no_selector")
            q_values_advantage = self._separate_advantage_network_structure(shared_dense,
                self.para_set.D_DENSE, num_action, memo="no_selector")
            q_values_advantage_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(q_values_advantage)
            bias = Subtract()([q_values_state, q_values_advantage_mean])
            q_values = Add(name="q_values")([bias, q_values_advantage])

        network = Model(inputs=[dic_input_node[feature_name]
                                for feature_name in self.feature_name_input[0:(3*num_neighbor)]],
                        outputs=q_values)
        network.compile(optimizer=RMSprop(lr=self.para_set.LEARNING_RATE),
                        loss="mean_squared_error")
        network.summary()

        return network

    def build_memory_separate(self):
        memory_list=[]
        for i in range(self.num_phases):
            memory_list.append([[] for j in range(self.num_actions)])
        return memory_list

    def remember_by_index(self, state, action, reward, next_state, p_index):
        self.memory[p_index].append([state, action, reward, next_state])

    # def clear_warm_up_memory(self):


    def remember(self, state, action, reward, next_state, p_index):

        if self.para_set.SEPARATE_MEMORY:
            ''' log the history separately '''
            self.memory[p_index][state.cur_phase[0][0]][action].append([state, action, reward, next_state])
        else:
            self.memory[p_index].append([state, action, reward, next_state])

    def remember_control(self, state, action, reward, next_state):

        if self.para_set.SEPARATE_MEMORY:
            ''' log the history separately '''
            self.control_memory[state.cur_phase[0][0]][action].append([state, action, reward, next_state])
        else:
            self.control_memory.append([state, action, reward, next_state])

    def forget(self, if_pretrain, net_index):

        if self.para_set.SEPARATE_MEMORY:
            ''' remove the old history if the memory is too large, in a separate way '''
            for phase_i in range(self.num_phases):
                for action_i in range(self.num_actions):
                    if if_pretrain:
                        random.shuffle(self.memory[net_index][phase_i][action_i])
                    if len(self.memory[net_index][phase_i][action_i]) > self.para_set.MAX_MEMORY_LEN:
                        print("length of memory (state {0}, action {1}): {2}, before forget".format(
                            phase_i, action_i, len(self.memory[net_index][phase_i][action_i])))
                        self.memory[net_index][phase_i][action_i] = self.memory[net_index][phase_i][action_i][-self.para_set.MAX_MEMORY_LEN:]
                    print("length of memory (state {0}, action {1}): {2}, after forget".format(
                        phase_i, action_i, len(self.memory[net_index][phase_i][action_i])))
        else:
            if len(self.memory[net_index]) > self.para_set.MAX_MEMORY_LEN:
                print("length of memory: {0}, before forget".format(len(self.memory[net_index])))
                self.memory[net_index] = self.memory[net_index][-self.para_set.MAX_MEMORY_LEN:]
            print("length of memory: {0}, after forget".format(len(self.memory[net_index])))

    def control_forget(self, if_pretrain):
        if self.para_set.SEPARATE_MEMORY:
            ''' remove the old history if the memory is too large, in a separate way '''
            for phase_i in range(self.num_phases):
                for action_i in range(self.num_actions):
                    if if_pretrain:
                        random.shuffle(self.control_memory[phase_i][action_i])
                    if len(self.control_memory[phase_i][action_i]) > self.para_set.MAX_MEMORY_LEN:
                        print("length of memory (state {0}, action {1}): {2}, before forget".format(
                            phase_i, action_i, len(self.control_memory[phase_i][action_i])))
                        self.control_memory[phase_i][action_i] = self.control_memory[phase_i][action_i][-self.para_set.MAX_MEMORY_LEN:]
                    print("length of memory (state {0}, action {1}): {2}, after forget".format(
                        phase_i, action_i, len(self.control_memory[phase_i][action_i])))
        else:
            if len(self.control_memory) > self.para_set.MAX_MEMORY_LEN:
                print("length of memory: {0}, before forget".format(len(self.control_memory)))
                self.control_memory = self.control_memory[-self.para_set.MAX_MEMORY_LEN:]
            print("length of memory: {0}, after forget".format(len(self.control_memory)))

    def _cal_average(self, sample_memory):

        list_reward = []
        average_reward = np.zeros((self.num_phases, self.num_actions))
        for phase_i in range(self.num_phases):
            list_reward.append([])
            for action_i in range(self.num_actions):
                list_reward[phase_i].append([])
        for [state, action, reward, _] in sample_memory:         # memory longer, slower ~~~
            phase = state.cur_phase[0][0]
            list_reward[phase][action].append(reward)

        for phase_i in range(self.num_phases):
            for action_i in range(self.num_actions):
                if len(list_reward[phase_i][action_i]) != 0:
                    average_reward[phase_i][action_i] = np.average(list_reward[phase_i][action_i])

        return average_reward

    def _cal_average_control(self, sample_memory):

        list_control_reward = []
        average_reward = np.zeros((self.control_num_actions))
        for action in range(self.control_num_actions):
            list_control_reward.append([])

        for [control_state, action, reward, _] in sample_memory:
            list_control_reward[action].append(reward)

        for action_i in range(self.control_num_actions):
            if len(list_control_reward[action_i]) != 0:
                average_reward[action_i] = np.average(list_control_reward[action_i])
        return average_reward

    def _cal_average_separate(self,sample_memory):
        ''' Calculate average rewards for different cases '''

        average_reward = np.zeros((self.num_phases, self.num_actions))
        for phase_i in range(self.num_phases):
            for action_i in range(self.num_actions):
                len_sample_memory = len(sample_memory[phase_i][action_i])
                if len_sample_memory > 0:
                    list_reward = []
                    for i in range(len_sample_memory):
                        state, action, reward, _ = sample_memory[phase_i][action_i][i]
                        list_reward.append(reward)
                    average_reward[phase_i][action_i]=np.average(list_reward)
        return average_reward


    def get_sample(self, memory_slice, gamma, net_index):

        len_memory_slice = len(memory_slice)
        # f_samples = open(os.path.join(self.path_set.PATH_TO_OUTPUT, "{0}_{1}_memory".format(net_index, prefix)), "a")
        State = []
        Action = []
        Reward = []
        Next_state = []
        for i in range(len_memory_slice):
            state, action, reward, next_state = memory_slice[i]
            State.append(state)
            Action.append(action)
            Reward.append(reward)
            Next_state.append(next_state)

        S_T_next= []
        len_feature = len(self.para_set.LIST_STATE_FEATURE)
        for i in range(len_feature):
            S_T_next.append([])
        for i in range(len(Next_state)):
            S_input_next = self.convert_state_to_input(Next_state[i])
            for j in range(len_feature):
                S_T_next[j].append(S_input_next[j][0])
        for j in range(len_feature):
            S_T_next[j]=np.array(S_T_next[j])
        a_max = np.argmax(self.q_network[net_index].predict(S_T_next), axis=1)

        next_estimated_reward = self.q_network_bar[net_index].predict(S_T_next)

        S_T= []
        for i in range(len_feature):
            S_T.append([])
        for i in range(len(State)):
            S_input = self.convert_state_to_input(State[i])
            for j in range(len_feature):
                S_T[j].append(S_input[j][0])
        for j in range(len_feature):
            S_T[j]=np.array(S_T[j])
        Tar = self.q_network[net_index].predict(S_T)
        Y_t = []
        for i in range(len(Tar)):
            Tar[i][Action[i]] = Reward[i] + gamma * next_estimated_reward[i][a_max[i]]
            Y_t.append(np.array(Tar[i]))

        return S_T, Y_t

    def get_control_sample(self, memory_slice, gamma):

        len_memory_slice = len(memory_slice)
        # f_samples_control = open(os.path.join(self.path_set.PATH_TO_OUTPUT, "{0}_control_memory".format(prefix)), "a")

        State = []
        Action = []
        Reward = []
        Next_state = []
        for i in range(len_memory_slice):
            control_state, action, reward, next_control_state = memory_slice[i]
            State.append(control_state)
            Action.append(action)
            Reward.append(reward)
            Next_state.append(next_control_state)

        len_feature = len(self.para_set.LIST_STATE_FEATURE)
        S_T_next= []
        for i in range(len_feature):
            S_T_next.append([])
        for i in range(len(Next_state)):
            S_input_next = self.convert_state_to_input(Next_state[i])
            for j in range(len_feature):
                S_T_next[j].append(S_input_next[j][0])
        for j in range(len_feature):
            S_T_next[j]=np.array(S_T_next[j])
        a_max = np.argmax(self.control_q_network.predict(S_T_next), axis=1)

        next_estimated_reward = self.control_q_network_bar.predict(S_T_next)

        S_T = []
        for i in range(len_feature):
            S_T.append([])
        for i in range(len(State)):
            S_input = self.convert_state_to_input(State[i])
            for j in range(len_feature):
                S_T[j].append(S_input[j][0])
        for j in range(len_feature):
            S_T[j]=np.array(S_T[j])
        Tar = self.control_q_network.predict(S_T)
        Y_t = []
        for i in range(len(Tar)):
            Tar[i][Action[i]] = Reward[i] + gamma * next_estimated_reward[i][a_max[i]]
            Y_t.append(np.array(Tar[i]))
            # f_samples_control.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(
            #     str(pre_control_target), str(control_target),
            #     str(control_action), str(control_reward), str(next_estimated_control_reward)
            # ))
        # f_samples_control.close()
        return S_T, Y_t

    def train_network(self, net_index, warm_up, Xs, Y, prefix, if_pretrain, current_time):

        if if_pretrain:
            epochs = self.para_set.EPOCHS_PRETRAIN
        else:
            epochs = self.para_set.EPOCHS
        batch_size = min(self.para_set.BATCH_SIZE, len(Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.para_set.PATIENCE, verbose=0, mode='min')

        hist = self.q_network[net_index].fit(Xs, Y, batch_size=batch_size, epochs=epochs,
                                shuffle=True,
                                verbose=0, validation_split=0.2, callbacks=[early_stopping])


    def update_network(self, net_index, warm_up, if_pretrain, use_average, current_time):
        ''' update Q network '''
        if warm_up:
            if current_time - self.update_outdated[net_index] < self.para_set.UPDATE_PERIOD:
                return

            self.update_outdated[net_index] = current_time

        # prepare the samples
        if if_pretrain:
            gamma = self.para_set.GAMMA_PRETRAIN
        else:
            gamma = self.para_set.GAMMA

        Y = []


        # get average state-action reward
        if self.memory[net_index] == []:
            return

        if self.para_set.SEPARATE_MEMORY:
            self.average_reward = self._cal_average_separate(self.memory[net_index])
        else:
            self.average_reward = self._cal_average(self.memory[net_index])

        # ================ sample memory ====================
        if self.para_set.SEPARATE_MEMORY:
            for phase_i in range(self.num_phases):
                for action_i in range(self.num_actions):
                    sampled_memory = self._sample_memory(
                        gamma=gamma,
                        with_priority=self.para_set.PRIORITY_SAMPLING,
                        memory=self.memory[phase_i][action_i],
                        if_pretrain=if_pretrain)
                    Xs, Y = self.get_sample(sampled_memory, gamma, net_index)
        else:
            sampled_memory = self._sample_memory(
                gamma=gamma,
                with_priority=self.para_set.PRIORITY_SAMPLING,
                memory=self.memory[net_index],
                if_pretrain=if_pretrain,
                net_index=net_index)

            Xs, Y = self.get_sample(sampled_memory, gamma, net_index)
        # ================ sample memory ====================

        Y = np.array(Y)

        # ============================  training  =======================================

        self.train_network(net_index, warm_up, Xs, Y, current_time, if_pretrain, current_time)

        self.q_bar_outdated[net_index] += 1
        self.forget(if_pretrain=if_pretrain, net_index=net_index)
        if net_index == 0 and self.para_set.EPSILON > 0.001:
            if warm_up:
                self.para_set.EPSILON = self.para_set.EPSILON * 0.98
            else:
                self.para_set.EPSILON = self.para_set.EPSILON * 0.97

    def train_control_network(self, Xs, Y, prefix, if_pretrain):

        if if_pretrain:
            epochs = self.para_set.EPOCHS_PRETRAIN
        else:
            epochs = self.para_set.EPOCHS
        batch_size = min(self.para_set.BATCH_SIZE, len(Y))

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.para_set.PATIENCE, verbose=0, mode='min')

        hist = self.control_q_network.fit(Xs, Y, batch_size=batch_size, epochs=epochs,
                                shuffle=True,
                                verbose=0, validation_split=0, callbacks=[early_stopping])
                                  # verbose=2, validation_split=0.3, callbacks=[early_stopping])


    def update_control_network(self, if_pretrain, use_average, current_time):
        ''' update Q network '''

        # prepare the samples
        if if_pretrain:
            gamma = self.para_set.GAMMA_PRETRAIN
        else:
            gamma = self.para_set.GAMMA

        # get average state-action reward
        self.average_reward = self._cal_average_control(self.control_memory)

        # ================ sample memory ====================
        sampled_memory = self._sample_memory_control(
            gamma=gamma,
            with_priority=self.para_set.PRIORITY_SAMPLING,
            memory=self.control_memory,
            if_pretrain=if_pretrain)

        Xs, Y = self.get_control_sample(sampled_memory, gamma)

        Y = np.array(Y)

        # ============================  training  =======================================
        self.train_control_network(Xs, Y, current_time, if_pretrain)
        self.control_q_bar_outdated += 1
        self.control_forget(if_pretrain=if_pretrain)

    def _sample_memory(self, gamma, with_priority, memory, if_pretrain, net_index):

        len_memory = len(memory)

        if not if_pretrain:
            sample_size = min(self.para_set.SAMPLE_SIZE, len_memory)
        else:
            sample_size = min(self.para_set.SAMPLE_SIZE_PRETRAIN, len_memory)

        if with_priority:
            # sample with priority
            sample_weight = []
            for i in range(len_memory):
                state, action, reward, next_state = memory[i]

                if state.if_terminal:
                    next_estimated_reward = 0
                else:
                    next_estimated_reward = self._get_next_estimated_reward(next_state, net_index)

                total_reward = reward + gamma * next_estimated_reward
                target = self.q_network[net_index].predict(
                    self.convert_state_to_input(state))
                
                pre_target = np.copy(target)
                # --- or ---
                # pre_reward = target[0][action]
                target[0][action] = total_reward

                # get the bias of current prediction
                weight = abs(pre_target[0][action] - total_reward)
                # --- or ---
                # weight = abs(pre_reward - total_reward)
                sample_weight.append(weight)

            priority = self._cal_priority(sample_weight)
            p = np.random.choice(range(len(priority)), sample_size, p=priority)
            p = np.unique(p)
            sampled_memory = np.array(memory)[p]
        else:
            sampled_memory = random.sample(memory, sample_size)

        return sampled_memory

    def _sample_memory_control(self, gamma, with_priority, memory, if_pretrain):

        len_memory = len(memory)

        if not if_pretrain:
            sample_size = min(self.para_set.SAMPLE_SIZE, len_memory)
        else:
            sample_size = min(self.para_set.SAMPLE_SIZE_PRETRAIN, len_memory)

        if with_priority:
            # sample with priority
            sample_weight = []
            for i in range(len_memory):
                state, action, reward, next_state = memory[i]


                next_estimated_reward = self._get_next_estimated_control_reward(next_state)

                total_reward = reward + gamma * next_estimated_reward
                target = self.control_q_network.predict(
                    self.convert_control_state_to_input(state))

                pre_target = np.copy(target)
                # --- or ---
                # pre_reward = target[0][action]
                target[0][action] = total_reward

                # get the bias of current prediction
                weight = abs(pre_target[0][action] - total_reward)
                # --- or ---
                # weight = abs(pre_reward - total_reward)
                sample_weight.append(weight)

            priority = self._cal_priority(sample_weight)
            p = np.random.choice(range(len(priority)), sample_size, p=priority)
            sampled_memory = np.array(memory)[p]
        else:
            sampled_memory = random.sample(memory, sample_size)

        return sampled_memory

    @staticmethod
    def _cal_priority(sample_weight):
        # pos_constant = 0.0001
        pos_constant = 0
        alpha = 1
        sample_weight_np = np.array(sample_weight)
        sample_weight_np = sample_weight_np / sample_weight_np.sum()
        # sample_weight_np = np.power(sample_weight_np + pos_constant, alpha) / sample_weight_np.sum()
        # sum_weight = sample_weight_np.sum()
        return sample_weight_np

class ValueNetwork():
    def __init__(self, num_features, hidden_size, learning_rate=.01):
        self.weight = 1
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.session = tf.Session()

            self.observations = tf.placeholder(shape=[None, self.num_features], dtype=tf.float32)
            self.W = [
                tf.get_variable("W1", shape=[self.num_features, self.hidden_size]),
                tf.get_variable("W2", shape=[self.hidden_size, self.hidden_size]),
                tf.get_variable("W3", shape=[self.hidden_size, 1])
            ]
            self.layer_1 = tf.nn.relu(tf.matmul(self.observations, self.W[0]))
            self.layer_2 = tf.nn.relu(tf.matmul(self.layer_1, self.W[1]))
            self.output = tf.reshape(tf.matmul(self.layer_2, self.W[2]), [-1])

            self.rollout = tf.placeholder(shape=[None], dtype=tf.float32)  # discounted_rewards
            self.loss = tf.losses.mean_squared_error(self.output, self.rollout)
            self.grad_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.trainable_vars = self.W
            self.get_grad = self.grad_optimizer.compute_gradients(self.loss, self.trainable_vars)
            self.minimize = self.grad_optimizer.minimize(self.loss)

            init = tf.global_variables_initializer()
            self.session.run(init)

    def get(self, states):
        value = self.session.run(self.output, feed_dict={self.observations: states})
        return value

    def update(self, states, discounted_rewards):
        gradient, _, loss = self.session.run([self.get_grad, self.minimize, self.loss], feed_dict={
            self.observations: states, self.rollout: discounted_rewards
        })
        # return gradient


class PPOPolicyNetwork():
    def __init__(self, num_features, layer_size, num_actions, epsilon=.2,
                 learning_rate=9e-4):
        self.weights = {}
        for node_id_i in map_computor.get_node_id_list():
            self.weights[node_id_i] = 1
        self.tf_graph = tf.Graph()

        with self.tf_graph.as_default():
            self.session = tf.Session()

            self.observations = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
            self.W = [
                tf.get_variable("W1", shape=[num_features, layer_size]),
                tf.get_variable("W2", shape=[layer_size, layer_size]),
                tf.get_variable("W3", shape=[layer_size, num_actions])
            ]
            # Add ops to save and restore all the variables.
            self.saver = tf.train.Saver(self.W, max_to_keep=1000)

            self.output = tf.nn.relu(tf.matmul(self.observations, self.W[0]))
            self.output = tf.nn.relu(tf.matmul(self.output, self.W[1]))
            self.output = tf.nn.softmax(tf.matmul(self.output, self.W[2]))

            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

            self.chosen_actions = tf.placeholder(shape=[None, num_actions], dtype=tf.float32)
            self.old_probabilities = tf.placeholder(shape=[None, num_actions], dtype=tf.float32)

            self.new_responsible_outputs = tf.reduce_sum(self.chosen_actions * self.output, axis=1)  # 降维，按行求和
            self.old_responsible_outputs = tf.reduce_sum(self.chosen_actions * self.old_probabilities, axis=1)

            self.ratio = self.new_responsible_outputs / self.old_responsible_outputs

            self.loss = tf.reshape(
                tf.minimum(
                    tf.multiply(self.ratio, self.advantages),
                    tf.multiply(tf.clip_by_value(self.ratio, 1 - epsilon, 1 + epsilon), self.advantages)),
                [-1]
            ) - 0.03 * self.new_responsible_outputs * tf.log(self.new_responsible_outputs + 1e-10)
            self.loss = -tf.reduce_mean(self.loss)

            self.W0_grad = tf.placeholder(dtype=tf.float32)
            self.W1_grad = tf.placeholder(dtype=tf.float32)
            self.W2_grad = tf.placeholder(dtype=tf.float32)

            self.gradient_placeholders = [self.W0_grad, self.W1_grad, self.W2_grad]
            self.trainable_vars = self.W
            self.gradients = [(np.zeros(var.get_shape()), var) for var in self.trainable_vars]
            # gradients:3 tuple,each tuple:len=2:1);梯度张量 2):对应的参数变量
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.get_grad = self.optimizer.compute_gradients(self.loss, self.trainable_vars)  # 计算梯度
            self.apply_grad = self.optimizer.apply_gradients(zip(self.gradient_placeholders,
                                                                 self.trainable_vars))  # 进行BP算法，由于apply_gradients函数接收的是一个(梯度张量, 变量)tuple列表，所以要将梯度列表和变量列表进行捉对组合,用zip函数
            init = tf.global_variables_initializer()
            self.session.run(init)

    def get_dist(self, states):
        dist = self.session.run(self.output, feed_dict={self.observations: states})
        return dist  # high level: dist = [[p1, p2, p3, p4]](选哪个子策略)	sub-policy: dist = [[p1, p2, p3, p4, p5]](怎么移动)

    def update(self, states, chosen_actions, ep_advantages):
        old_probabilities = self.session.run(self.output, feed_dict={self.observations: states})
        self.session.run(self.apply_grad, feed_dict={
            self.W0_grad: self.gradients[0][0],
            self.W1_grad: self.gradients[1][0],
            self.W2_grad: self.gradients[2][0],

        })
        self.gradients, loss = self.session.run([self.get_grad, self.output], feed_dict={
            self.observations: states,
            self.advantages: ep_advantages,
            self.chosen_actions: chosen_actions,
            self.old_probabilities: old_probabilities
        })

    def get_gradient(self, states, chosen_actions, ep_advantages):
        old_probabilities = self.session.run(self.output, feed_dict={self.observations: states})
        gradients = self.session.run(self.get_grad, feed_dict={
            self.observations: states,
            self.advantages: ep_advantages,
            self.chosen_actions: chosen_actions,
            self.old_probabilities: old_probabilities
        })
        return gradients

    def update_weight(self, gradient_main, gradient_aux, node_id):
        alpha = 0.0003
        beita = 0.001
        gradient_main = np.array(gradient_main)
        gradient_aux = np.array(gradient_aux)
        gradient_dot = gradient_main * gradient_aux
        m1 = list(gradient_dot)
        dot_sum = 0
        for m2 in m1:
            m2 = list(m2)
            for m3 in m2:
                m3 = list(m3)
                for m4 in m3:
                    m4 = list(m4)
                    dot_sum += sum(m4)
        dot_sum1 = alpha * dot_sum

        self.weights[node_id] -= beita * dot_sum1
        return self.weights[node_id], dot_sum

    def save_w(self, name):
        self.saver.save(self.session, name + '.ckpt')

    def restore_w(self, name):
        self.saver.restore(self.session, name + '.ckpt')