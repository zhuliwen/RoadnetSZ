# -*- coding: utf-8 -*-

'''

Network agent from agent.py

'''


import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Multiply, Add
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
import random
from keras.engine.topology import Layer
import os

from agent import Agent, State

class Selector(Layer):

    def __init__(self, select, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.select = select
        self.select_neuron = K.constant(value=self.select)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Selector, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.cast(K.equal(x, self.select_neuron), dtype="float32")

    def get_config(self):
        config = {"select": self.select}
        base_config = super(Selector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def conv2d_bn(input_layer, index_layer,
              filters=16,
              kernel_size=(3, 3),
              strides=(1, 1)):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  use_bias=False,
                  name="conv{0}".format(index_layer))(input_layer)
    bn = BatchNormalization(axis=bn_axis, scale=False, name="bn{0}".format(index_layer))(conv)
    act = Activation('relu', name="act{0}".format(index_layer))(bn)
    # pooling = MaxPooling2D(pool_size=2)(act)
    # x = Dropout(0.3)(pooling)
    # return x
    return act


class NetworkAgent(Agent):

    @staticmethod
    def _unison_shuffled_copies(Xs, Y, sample_weight):
        p = np.random.permutation(len(Y))
        new_Xs = []
        for x in Xs:
            assert len(x) == len(Y)
            new_Xs.append(x[p])
        return new_Xs, Y[p], sample_weight[p]

    @staticmethod
    # --- my modification ---
    def _cnn_network_structure(img_features):
        # 8, 150,150 -> 32, 150/5=30, 150/5=30
        conv1 = conv2d_bn(img_features, 1, filters=32, kernel_size=(5, 5), strides=(5, 5))
        # 32, 30, 30 -> 64, 30/3=10, 30/3=10
        conv2 = conv2d_bn(conv1, 2, filters=64, kernel_size=(3, 3), strides=(3, 3))
        # 64, 10, 10 -> 2, 10, 10
        conv3 = conv2d_bn(conv2, 3, filters=2, kernel_size=(1, 1), strides=(1, 1))
        img_flatten = Flatten()(conv3)
        return img_flatten

    @staticmethod
    def _shared_network_structure(state_features, dense_d):
        hidden_1 = Dense(dense_d, activation="sigmoid", name="hidden_shared_1")(state_features)
        return hidden_1

    @staticmethod
    def _separate_advantage_network_structure(state_features, dense_d, num_actions, memo=""):
        # hidden_1 = Dense(dense_d, activation="sigmoid", name="hidden_separate_branch_{0}_1".format(memo))(state_features)
        hidden_1 = Dense(dense_d, activation="relu", name="hidden_separate_advantage_branch_{0}_1".format(memo))(state_features)
        q_values = Dense(num_actions, activation="linear", name="q_values_separate_advantage_branch_{0}".format(memo))(hidden_1)
        return q_values

    @staticmethod
    def _separate_state_network_structure(state_features, dense_d, memo=""):
        # hidden_1 = Dense(dense_d, activation="sigmoid", name="hidden_separate_branch_{0}_1".format(memo))(state_features)
        hidden_1 = Dense(dense_d, activation="relu", name="hidden_separate_state_branch_{0}_1".format(memo))(state_features)
        v_value = Dense(1, activation="linear", name="q_values_separate_state_branch_{0}".format(memo))(hidden_1)
        return v_value



    def load_model(self, path_to_model, file_name, layer_dict, network_index):
        # --- my modification ---
        # self.q_network = load_model(os.path.join(self.path_set.PATH_TO_MODEL, "%s_q_network.h5" % file_name))
        self.q_network.append(load_model(os.path.join(path_to_model, file_name), layer_dict))

    def load_control_model(self, path_to_model, file_name, layer_dict):
        # --- my modification ---
        # self.q_network = load_model(os.path.join(self.path_set.PATH_TO_MODEL, "%s_q_network.h5" % file_name))
        self.control_q_network = load_model(os.path.join(path_to_model, file_name), layer_dict)

    def save_model(self, episode, network_index):
        self.q_network[network_index].save(os.path.join(self.path_set.PATH_TO_MODEL, "%s_%d.h5" % (episode, network_index)))

    def save_control_model(self, episode):
        self.control_q_network.save(os.path.join(self.path_set.PATH_TO_MODEL, "%s_control.h5" % (episode)))


    def choose(self, count, if_pretrain, p_index):

        ''' choose the best action for current state '''

        q_values = self.q_network[p_index].predict(self.convert_state_to_input(self.state))
        # print(q_values)
        if if_pretrain:
            self.action = np.argmax(q_values[0])
        else:
            if random.random() <= self.para_set.EPSILON:  # continue explore new Random Action
                self.action = random.randrange(len(q_values[0]))
                print("##Explore")
            else:  # exploitation
                self.action = np.argmax(q_values[0])
        return self.action, q_values

    def control_choose(self, count, if_pretrain):

        ''' choose the best action for current state '''

        q_values = self.control_q_network.predict(self.convert_state_to_input(self.state))
        # print(q_values)
        if if_pretrain:
            self.control_action = np.argmax(q_values[0])
        else:
            if random.random() <= self.para_set.EPSILON:  # continue explore new Random Action
                self.control_action = random.randrange(len(q_values[0]))
                print("##Explore")
            else:  # exploitation
                self.control_action = np.argmax(q_values[0])

        return self.control_action, q_values

    def build_memory(self):

        return [[],[],[]]
    def control_build_memory(self):

        return []

    #
    # def batch_predict(self,file_name="temp"):
    #     f_samples = open("./records/DQN_v1/" + file_name[:file_name.rfind("_")] + "predict_pretrain.txt", "a")
    #     f_samples_head = ["state.cur_phase", "state.time_this_phase",
    #                       "target",
    #                       "action",
    #                       "reward"]
    #     f_samples.write('\t'.join(f_samples_head) + "\n")
    #     len_memory = len(self.memory)
    #     for i in range(len_memory):
    #         state, action, reward, next_state = self.memory[i]
    #         q_values = self.q_network.predict(self.convert_state_to_input(state))
    #         f_samples.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(
    #             str(state.cur_phase[0]), str(state.time_this_phase), str(q_values),
    #             str(action), str(reward)
    #         ))

    def remember_by_index(self, state, action, reward, next_state, p_index):
        self.memory[p_index].append([state, action, reward, next_state])

    def remember(self, state, action, reward, next_state, p_index):

        ''' log the history '''
        self.memory.append([state, action, reward, next_state])

    def clear_memory(self):
        memory = [[], [], []]
        return memory

    def forget(self):

        ''' remove the old history if the memory is too large '''

        if len(self.memory) > self.para_set.MAX_MEMORY_LEN:
            print("length of memory: {0}, before forget".format(len(self.memory)))
            self.memory = self.memory[-self.para_set.MAX_MEMORY_LEN:]
            print("length of memory: {0}, after forget".format(len(self.memory)))

    def _get_next_estimated_reward(self, next_state, net_index):

        if self.para_set.DDQN:
            a_max = np.argmax(self.q_network[net_index].predict(
                self.convert_state_to_input(next_state))[0])
            next_estimated_reward = self.q_network_bar[net_index].predict(
                self.convert_state_to_input(next_state))[0][a_max]
            return next_estimated_reward
        else:
            next_estimated_reward = np.max(self.q_network_bar[net_index].predict(
                self.convert_state_to_input(next_state))[0])
            return next_estimated_reward

    def _get_next_estimated_control_reward(self, next_control_state):

        if self.para_set.DDQN:
            a_max = np.argmax(self.control_q_network.predict(
                self.convert_control_state_to_input(next_control_state))[0])
            next_estimated_reward = self.control_q_network_bar.predict(
                self.convert_control_state_to_input(next_control_state))[0][a_max]
            return next_estimated_reward
        else:
            next_estimated_reward = np.max(self.control_q_network_bar.predict(
                self.convert_control_state_to_input(next_control_state))[0])
            return next_estimated_reward

    def update_control_network_bar(self):

        ''' update Q bar '''

        if self.control_q_bar_outdated >= self.para_set.UPDATE_CONTROL_Q_BAR_FREQ:
            self.control_q_network_bar.set_weights(self.control_q_network.get_weights())
            self.control_q_bar_outdated = 0

    def update_network_bar(self, net_index):

        ''' update Q bar '''

        if self.q_bar_outdated[net_index] >= self.para_set.UPDATE_Q_BAR_FREQ:
            self.q_network_bar[net_index].set_weights(self.q_network[net_index].get_weights())
            self.q_bar_outdated[net_index] = 0


