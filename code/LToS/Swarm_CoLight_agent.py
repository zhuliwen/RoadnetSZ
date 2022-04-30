import numpy as np 
import os 
# import pickle  
# from agent import Agent
from CoLight_agent import CoLightAgent, RepeatVector3D
from noise import OrnsteinUhlenbeckActionNoise
from l2_relu_regularizer import L2ReLURegularizer
import random 
import time
"""
Model for my model: SwarmCoLight
"""
# import keras
from keras import backend as K
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
from keras.layers import Dense, Dropout, Conv2D, Input, Lambda, Flatten, TimeDistributed, merge
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.models import Model, model_from_json, load_model
from keras.layers.core import Activation
# from keras.utils import np_utils,to_categorical
# from keras.engine.topology import Layer
# from keras.callbacks import EarlyStopping, TensorBoard

class SwarmCoLightAgent(CoLightAgent): 
    def __init__(self, 
        dic_agent_conf=None, 
        dic_traffic_env_conf=None, 
        dic_path=None, 
        cnt_round=None, 
        best_round=None, bar_round=None,inter_info=None,intersection_id="0"):
        """
        #1. compute the (dynamic) static Adjacency matrix, compute for each state
        -2. #neighbors: 5 (1 itself + W,E,S,N directions)
        -3. compute len_features
        -4. self.num_actions
        """
        # w_in
        # we need that to call build_network() before calling of super initialization
        self.inter_info = inter_info
        self.num_in_neighbors = len(inter_info[0].in_adjacency_row)

        self.lr = dic_agent_conf["LEARNING_RATE"] * (dic_agent_conf["LEARNING_RATE_DECAY"]**cnt_round)
        self.w_lr = dic_agent_conf["W_LEARNING_RATE"] * (dic_agent_conf["W_LEARNING_RATE_DECAY"]**cnt_round)
        self.patience = int(dic_agent_conf["PATIENCE"] * (dic_agent_conf["PATIENCE_DECAY"]**cnt_round)) # >= 0
        self.w_epochs = max(int(dic_agent_conf["W_EPOCHS"] * (dic_agent_conf["W_EPOCHS_DECAY"]**cnt_round)), 1)
        
        super(SwarmCoLightAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round,
            best_round, bar_round, intersection_id)

        adjacency_rows = [inter.adjacency_row for inter in inter_info]
        w_out_mask = np.array(adjacency_rows, dtype=np.int)
        w_out_mask[w_out_mask==-1] = 1e8
        w_out_mask = np.sort(w_out_mask,axis=-1) # 0,1,2,...,1e8,1e8,...

        self_mask = np.zeros_like(w_out_mask, dtype=np.float32)
        self_indice = [np.argwhere(w_out_mask[i,:] == i) for i in range(self.num_agents)]
        
        self.v_grad_mask = np.copy(w_out_mask)
        
        # padding -40 as mask for softmax
        w_out_mask[w_out_mask!=1e8] = 0
        w_out_mask[w_out_mask==1e8] = -40 # 0,0,0,...,-40,-40,...
        # initialize with self's w bigger
        own_ratio = self.dic_agent_conf["INITIAL_OWN_W_OUT_RATIO"]
        for i in range(self.num_agents):
            real_num_out_neighbors = len(np.where(w_out_mask[i,:] == 0)[0])
            if real_num_out_neighbors > 1:
                self_mask[i,self_indice[i]] = np.log(own_ratio/(1-own_ratio)*(real_num_out_neighbors-1))
        # padding 0 as mask for multiply()
        self.v_grad_mask[self.v_grad_mask!=1e8] = 1
        self.v_grad_mask[self.v_grad_mask==1e8] = 0 # 1,1,1,...,0,0,...

        # build v network for w optimization
        self.v_network = self.build_v_network(bar=False)
        self.v_network_bar = self.build_v_network(bar=True)
        self.sess = K.get_session()
        self.get_v_grad = self.v_gradient()

        if cnt_round == 0: 
            # initialization
            # self.w_network = self.build_w_network(w_out_mask, self_mask)
            # if os.listdir(self.dic_path["PATH_TO_MODEL"]):
            #     self.w_network.load_weights(
            #         os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_{0}_w.h5".format(intersection_id)), 
            #         by_name=True)
            # self.w_network_bar = self.build_network_from_copy(self.w_network)
            if self.dic_path["PATH_TO_RELOAD_MODEL"] is not None and os.listdir(self.dic_path["PATH_TO_RELOAD_MODEL"]):
                file_path = self.dic_path["PATH_TO_RELOAD_MODEL"]
                file_name = "inter_{0}_w".format(self.intersection_id)
                self.w_network = load_model(
                    os.path.join(file_path, "%s.h5" % file_name),
                    custom_objects={'RepeatVector3D':RepeatVector3D, 'L2ReLURegularizer':L2ReLURegularizer})
                file_name_bar = "inter_{0}_w_bar".format(self.intersection_id)
                self.w_network_bar = load_model(
                    os.path.join(file_path, "%s.h5" % file_name_bar),
                    custom_objects={'RepeatVector3D':RepeatVector3D, 'L2ReLURegularizer':L2ReLURegularizer})
                print("succeed in reloading model for w_network initialization from %s"%self.dic_path["PATH_TO_RELOAD_MODEL"])
            else:
                self.w_network = self.build_w_network(w_out_mask, self_mask)
                # self.w_network_bar = self.build_network_from_copy(self.w_network)
                self.w_network_bar = self.build_w_network(w_out_mask, self_mask)
                self.w_network_bar.set_weights(self.w_network.get_weights())
        else:
            try:
                if best_round:
                    # use model pool
                    # self.load_w_network("round_{0}_inter_{1}_w_bar".format(best_round,self.intersection_id))
                    self.load_w_network("round_{0}_inter_{1}_w_bar".format(best_round,self.intersection_id), file_path=None,
                        w_out_mask=w_out_mask, self_mask=self_mask)

                    if bar_round and bar_round != best_round and cnt_round > 10:
                        # load w_bar network from model pool
                        # self.load_w_network_bar("round_{0}_inter_{1}_w_bar".format(bar_round,self.intersection_id))
                        self.load_w_network_bar("round_{0}_inter_{1}_w_bar".format(bar_round,self.intersection_id), file_path=None,
                            w_out_mask=w_out_mask, self_mask=self_mask)
                    else:
                        if "UPDATE_W_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                            if self.dic_agent_conf["UPDATE_W_BAR_EVERY_C_ROUND"]:
                                # self.load_w_network_bar("round_{0}".format(
                                #     max((best_round - 1) // self.dic_agent_conf["UPDATE_W_BAR_FREQ"] * self.dic_agent_conf["UPDATE_W_BAR_FREQ"], 0),
                                #     self.intersection_id))
                                self.load_w_network_bar("round_{0}".format(
                                    max((best_round - 1) // self.dic_agent_conf["UPDATE_W_BAR_FREQ"] * self.dic_agent_conf["UPDATE_W_BAR_FREQ"], 0),
                                    self.intersection_id), file_path=None, w_out_mask=w_out_mask, self_mask=self_mask)
                            else:
                                # self.load_w_network_bar("round_{0}_inter_{1}_w_bar".format(
                                #     max(best_round - self.dic_agent_conf["UPDATE_W_BAR_FREQ"], 0),
                                #     self.intersection_id))
                                self.load_w_network_bar("round_{0}_inter_{1}_w_bar".format(
                                    max(best_round - self.dic_agent_conf["UPDATE_W_BAR_FREQ"], 0),
                                    self.intersection_id), file_path=None, w_out_mask=w_out_mask, self_mask=self_mask)
                        else:
                            # self.load_w_network_bar("round_{0}_inter_{1}_w_bar".format(
                            #     max(best_round - self.dic_agent_conf["UPDATE_W_BAR_FREQ"], 0), self.intersection_id))
                            self.load_w_network_bar("round_{0}_inter_{1}_w_bar".format(
                                max(best_round - self.dic_agent_conf["UPDATE_W_BAR_FREQ"], 0), self.intersection_id),
                                file_path=None, w_out_mask=w_out_mask, self_mask=self_mask)

                else:
                    # not use model pool
                    #TODO how to load network for multiple intersections?
                    # print('init q load')
                    # self.load_w_network("round_{0}_inter_{1}_w_bar".format(cnt_round-1, self.intersection_id))
                    self.load_w_network("round_{0}_inter_{1}_w_bar".format(cnt_round-1, self.intersection_id), file_path=None,
                        w_out_mask=w_out_mask, self_mask=self_mask)
                    # print('init q_bar load')
                    if "UPDATE_W_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                        if self.dic_agent_conf["UPDATE_W_BAR_EVERY_C_ROUND"]:
                            # self.load_w_network_bar("round_{0}_inter_{1}_w_bar".format(
                            #     max((cnt_round - 1) // self.dic_agent_conf["UPDATE_W_BAR_FREQ"] * self.dic_agent_conf["UPDATE_W_BAR_FREQ"], 0),
                            #     self.intersection_id))
                            self.load_w_network_bar("round_{0}_inter_{1}_w_bar".format(
                                max((cnt_round - 1) // self.dic_agent_conf["UPDATE_W_BAR_FREQ"] * self.dic_agent_conf["UPDATE_W_BAR_FREQ"], 0),
                                self.intersection_id), file_path=None, w_out_mask=w_out_mask, self_mask=self_mask)
                        else:
                            # self.load_w_network_bar("round_{0}_inter_{1}_w_bar".format(
                            #     max(cnt_round - self.dic_agent_conf["UPDATE_W_BAR_FREQ"], 0),
                            #     self.intersection_id))
                            self.load_w_network_bar("round_{0}_inter_{1}_w_bar".format(
                                max(cnt_round - self.dic_agent_conf["UPDATE_W_BAR_FREQ"], 0),
                                self.intersection_id), file_path=None, w_out_mask=w_out_mask, self_mask=self_mask)
                    else:
                        # self.load_w_network_bar("round_{0}_inter_{1}_w_bar".format(
                        #     max(cnt_round - self.dic_agent_conf["UPDATE_W_BAR_FREQ"], 0), self.intersection_id))
                        self.load_w_network_bar("round_{0}_inter_{1}_w_bar".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_W_BAR_FREQ"], 0), self.intersection_id),
                            file_path=None, w_out_mask=w_out_mask, self_mask=self_mask)
            except:
                print("fail to load network, current round: {0}".format(cnt_round))
                exstr = traceback.format_exc()
                print(exstr)

        # init w
        self.w_optimizer()
        # if training and it's time to update w
        if self.dic_agent_conf["EPSILON"] != 0 and (cnt_round+1) % self.dic_agent_conf['UPDATE_W_FREQ'] == 0:
            # for w training and exploring:
            sigma = self.dic_agent_conf["W_SIGMA"]
            mu = sigma*np.random.normal(size=(1,self.num_agents,self.num_neighbors))
            self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=mu, sigma=sigma, x0=mu)
        else:
            self.ou_noise = None

    def w_out_to_w_in(self, w_out, raw_total_adjs, adjs=None):
        """
        w_out: [batch,agents,neighbors]
        raw_total_adjs: [batch,agents,neighbors]
        """

        # anon_env.py#L1300+: if ADJACENCY_BY_CONNECTION_OR_GEO, there's a guarantee that top-k neighbors exist
        # else, some neighbors may not be there and their indice are all -1
        batch_size, num_agents, num_neighbors = w_out.shape
        full_w_matrix = -np.ones((batch_size,num_agents,num_agents))
        if adjs is None:
            raw_total_adjs[raw_total_adjs==-1] = 1e8
            raw_total_adjs=np.sort(raw_total_adjs,axis=-1)
            raw_total_adjs[raw_total_adjs==1e8] = -1
            # still put all -1 at the tail
            for i in range(batch_size):
                for j in range(num_agents):
                    # out_neighbors = w_out[i,j,:] > 0 # or use np.where
                    # full_w_matrix[i,j,raw_total_adjs[i,j,out_neighbors]] = w_out[i,j,out_neighbors]
                    out_neighbors = raw_total_adjs[i,j,raw_total_adjs[i,j,:] != -1]
                    full_w_matrix[i,j,out_neighbors] = w_out[i,j,:len(out_neighbors)]
                    # illegal places remain -1
        else: # use adj
            # to_categorical back to label
            out_neighbor_flag = np.sum(adjs, axis=2)
            for i in range(batch_size):
                for j in range(num_agents):
                    out_neighbors = np.where(out_neighbor_flag[i,j,:] == 1)[0]
                    full_w_matrix[i,j,out_neighbors] = w_out[i,j,:len(out_neighbors)]
                
        full_w_matrix = np.transpose(full_w_matrix, (0,2,1))
        w_in = np.zeros((batch_size,num_agents,self.num_in_neighbors))
        for i in range(batch_size):
            for j in range(num_agents):
                # in-neighbors might be more than limited
                in_neighbors = np.where(full_w_matrix[i,j,:] >= 0)[0]
                w_in[i,j,:len(in_neighbors)] = full_w_matrix[i,j,in_neighbors]

        return w_in

    def w_in_att_predict(self,state,total_features=[],total_adjs=[],bar=False):
        #state:[batch,agent,features and adj]
        #return:act:[batch,agent],att:[batch,layers,agent,head,neighbors]
        batch_size=len(state)
        if total_features==[] and total_adjs==[]:
            total_features,total_adjs=list(),list()
            for i in range(batch_size): 
                feature=[]
                adj=[] 
                for j in range(self.num_agents):
                    observation=[]
                    for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                        if 'adjacency' in feature_name:
                            continue
                        if feature_name == "cur_phase":
                            if len(state[i][j][feature_name])==1:
                                #choose_action
                                observation.extend(self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']]
                                                            [state[i][j][feature_name][0]])
                            else:
                                observation.extend(state[i][j][feature_name])
                        elif feature_name=="lane_num_vehicle":
                            observation.extend(state[i][j][feature_name])
                    feature.append(observation)
                    adj.append(state[i][j]['adjacency_matrix'])
                total_features.append(feature)
                total_adjs.append(adj)
            #feature:[agents,feature]
            total_features=np.reshape(np.array(total_features),[batch_size,self.num_agents,-1])
            raw_total_adjs=np.array(total_adjs, dtype=np.int)
            total_adjs=self.adjacency_index2matrix(np.array(total_adjs))
            #adj:[agent,neighbors]   

        #out: [batch,agent,neighbors], att:[batch,layers,agent,head,neighbors]
        # softmax
        def softmax(x):
            shifted_x = x - np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(shifted_x)
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        if batch_size>1:
            # TD3 add noise
            if bar:
                w_out_before, attention= self.w_network_bar.predict([total_features,total_adjs])
            else:
                w_out_before, attention= self.w_network.predict([total_features,total_adjs])
            # if self.ou_noise is not None:
            if True:
                noise = np.random.normal(size=w_out_before.shape) * 0.1
                w_out = softmax(w_out_before + noise)
            else:
                w_out = softmax(w_out_before)
            w_in = self.w_out_to_w_in(w_out, raw_total_adjs)
            
            return total_features,total_adjs,w_in,attention
        else:
            # explore with noise
            if bar:
                w_out_before, attention= self.w_network_bar.predict([total_features,total_adjs])
            else:
                w_out_before, attention= self.w_network.predict([total_features,total_adjs])
            if self.ou_noise is not None:
                noise = self.ou_noise() * self.dic_agent_conf["EPSILON"]
                w_out = softmax(w_out_before + noise)
            else:
                w_out = softmax(w_out_before)
            w_in = self.w_out_to_w_in(w_out, raw_total_adjs)

            return w_in,attention

    def action_att_predict(self,state,w_in,total_features=[],total_adjs=[],bar=False):
        #state:[batch,agent,features+w_in and adj]
        #return:act:[batch,agent],att:[batch,layers,agent,head,neighbors]
        batch_size=len(state)
        if total_features==[] and total_adjs==[]:
            total_features,total_ws,total_adjs=list(),list(),list()
            for i in range(batch_size): 
                feature=[]
                adj=[] 
                for j in range(self.num_agents):
                    observation=[]
                    for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                        if 'adjacency' in feature_name:
                            continue
                        if feature_name == "cur_phase":
                            if len(state[i][j][feature_name])==1:
                                #choose_action
                                observation.extend(self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']]
                                                            [state[i][j][feature_name][0]])
                            else:
                                observation.extend(state[i][j][feature_name])
                        elif feature_name=="lane_num_vehicle":
                            observation.extend(state[i][j][feature_name])
                    feature.append(observation)
                    adj.append(state[i][j]['adjacency_matrix'])
                total_features.append(feature)
                total_adjs.append(adj)
            #feature:[agents,feature]
            total_features=np.reshape(np.array(total_features),[batch_size,self.num_agents,-1])
            total_adjs=self.adjacency_index2matrix(np.array(total_adjs))
            #adj:[agent,neighbors]   
        total_w_in = np.array(w_in)
        if bar:
            all_output= self.q_network_bar.predict([total_features,total_w_in,total_adjs])
        else:
            all_output= self.q_network.predict([total_features,total_w_in,total_adjs])
        action,attention =all_output[0],all_output[1]

        #out: [batch,agent,action], att:[batch,layers,agent,head,neighbors]
        if len(action)>1:
            return total_features,total_w_in,total_adjs,action,attention

        #[batch,agent,1]
        max_action=np.expand_dims(np.argmax(action,axis=-1),axis=-1)
        random_action=np.reshape(np.random.randint(self.num_actions,size=1*self.num_agents),(1,self.num_agents,1))
        #[batch,agent,2]
        possible_action=np.concatenate([max_action,random_action],axis=-1)
        selection=np.random.choice(
            [0,1],
            size=batch_size*self.num_agents,
            p=[1-self.dic_agent_conf["EPSILON"],self.dic_agent_conf["EPSILON"]])
        act=possible_action.reshape((batch_size*self.num_agents,2))[np.arange(batch_size*self.num_agents),selection]
        act=np.reshape(act,(batch_size,self.num_agents))
        return act,attention

    def choose_w_in(self, count, state):

        ''' 
        choose the best w_in for current state 
        -input: state: [batch,agent,feature]  adj: [batch,agent,neighbors,agents]
        -output: out: [batch,agent,in_neighbors], att:[batch,layers,agent,head,neighbors]
        '''
        w_in,attention=self.w_in_att_predict([state])
        return w_in[0],attention[0]

    def choose_action(self, count, state, w_in):

        ''' 
        choose the best action for current state 
        -input: state: [batch,agent,feature]  adj: [batch,agent,neighbors,agents]
        -output: out: [batch,agent,action], att:[batch,layers,agent,head,neighbors]
        '''
        act,attention=self.action_att_predict([state], [w_in])
        return act[0],attention[0]

    def prepare_Xs_Y(self, memory, dic_exp_conf):
        """
        
        """
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting
        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            sample_slice = memory
        # forget
        else:
            ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
            memory_after_forget = memory[ind_sta: ind_end]
            print("memory size after forget:", len(memory_after_forget))

            # sample the memory
            sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
            sample_slice = random.sample(memory_after_forget, sample_size)
            print("memory samples number:", sample_size)

        _state = []
        _next_state = []
        _w_in=[]
        _action=[]
        _reward=[]

        for i in range(len(sample_slice)):  
            _state.append([])
            _next_state.append([])
            _w_in.append([])
            _action.append([])
            _reward.append([])
            for j in range(self.num_agents):
                state, w_in, action, next_state, reward, _ = sample_slice[i][j]
                _state[i].append(state)
                _next_state[i].append(next_state)
                _w_in[i].append(w_in)
                _action[i].append(action)
                _reward[i].append(reward)


        #target: [#agents,#samples,#num_actions]    
        _features,_w_in,_adjs,q_values,_=self.action_att_predict(_state,_w_in)   
        _,_,_next_w_in,_ = self.w_in_att_predict(
            _next_state,
            bar=True)
        _next_features,_next_w_in,_next_adjs,_,attention= self.action_att_predict(_next_state,_next_w_in)
        #target_q_values:[batch,agent,action]
        _,_,_,target_q_values,_= self.action_att_predict(
            _next_state,_next_w_in,
            total_features=_next_features,
            total_adjs=_next_adjs,
            bar=True)

        for i in range(len(sample_slice)):
            for j in range(self.num_agents):
                q_values[i][j][_action[i][j]] = _reward[i][j]/ self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(target_q_values[i][j])


        #self.Xs should be: [#agents,#samples,#features+#]
        self.Xs = [_features,_w_in,_adjs]
        self.Y=q_values.copy()
        self.Y_total = [q_values.copy()]
        self.Y_total.append(attention)
        return  

    def build_w_network(
        self,
        w_out_mask, self_mask,
        MLP_layers=[32,32], 
        # CNN_layers=[[32,32]],#[[4,32],[4,32]],
        # CNN_heads=[1],#[8,8],
        Output_layers=[]):
        CNN_layers=self.CNN_layers 
        CNN_heads=[1]*len(CNN_layers)
        """
        layer definition
        """
        start_time=time.time()
        assert len(CNN_layers)==len(CNN_heads)

        In=list()
        """
        #[#agents,batch,feature_dim],[#agents,batch,neighbors,agents],[batch,1,neighbors]
        ->[#agentsxbatch,feature_dim],[#agentsxbatch,neighbors,agents],[batch,1,neighbors]
        """
        #In: [batch,agent,feature]
        #In: [batch,agent,neighbors,agents]
        In.append(Input(shape=[self.num_agents,self.len_feature],name="feature"))
        In.append(Input(shape=(self.num_agents,self.num_neighbors,self.num_agents),name="in_adjacency_matrix"))


        Input_end_time=time.time()
        """
        Currently, the MLP layer 
        -input: [batch,agent,feature_dim]
        -outpout: [#agent,batch,128]
        """
        feature=self.MLP(In[0],MLP_layers)

        Embedding_end_time=time.time()


        #TODO: remove the dense setting
        #feature:[batch,agents,feature_dim]
        att_record_all_layers=list()
        print("CNN_heads:", CNN_heads)
        for CNN_layer_index,CNN_layer_size in enumerate(CNN_layers):
            print("CNN_heads[CNN_layer_index]:",CNN_heads[CNN_layer_index])
            if CNN_layer_index==0:
                h,att_record=self.MultiHeadsAttModel(
                    feature,
                    In[1],
                    l=self.num_neighbors,
                    d=MLP_layers[-1],
                    dv=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    nv=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                    )
            else:
                h,att_record=self.MultiHeadsAttModel(
                    h,
                    In[1],
                    l=self.num_neighbors,
                    d=MLP_layers[-1],
                    dv=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    nv=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                    )
            att_record_all_layers.append(att_record)

        if len(CNN_layers)>1:
            att_record_all_layers=Concatenate(axis=1)(att_record_all_layers)
        else:
            att_record_all_layers=att_record_all_layers[0]

        att_record_all_layers=Reshape(
            (len(CNN_layers),self.num_agents,CNN_heads[-1],self.num_neighbors), name='attention'
            )(att_record_all_layers)

        
        #TODO remove dense net
        for layer_index,layer_size in enumerate(Output_layers):
                h=Dense(layer_size,activation='relu',kernel_initializer='random_normal',name='Dense_q_%d'%layer_index)(h)
        #action prediction layer
        #[batch,agent,32]->[batch,agent,neighbors]
        out = Dense(self.num_neighbors,kernel_initializer='random_normal',name='unmasked_w_before_softmax')(h)
        out = Lambda(lambda x: x+w_out_mask+self_mask, name='masked_w_before_softmax')(out)
        # no softmax
        #out:[batch,agent,action], att:[batch,layers,agent,head,neighbors]
        model=Model(inputs=In,outputs=[out,att_record_all_layers])

        if self.att_regulatization:
            model.compile(
                optimizer=RMSprop(lr=self.lr),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"],'kullback_leibler_divergence'],
                loss_weights=[1,self.dic_agent_conf["rularization_rate"]])
        else:
            model.compile(
                optimizer=RMSprop(lr=self.lr),
                loss=self.dic_agent_conf["LOSS_FUNCTION"],
                loss_weights=[1,0])
        # model.compile(optimizer=Adam(lr = 0.0001), loss='mse')
        model.summary()
        network_end=time.time()
        print('build_Input_end_time：',Input_end_time-start_time)
        print('embedding_time:',Embedding_end_time-Input_end_time)
        print('total time:',network_end-start_time)
        return model

    def build_network(
        self,
        MLP_layers=[32,32], 
        # CNN_layers=[[32,32]],#[[4,32],[4,32]],
        # CNN_heads=[1],#[8,8],
        Output_layers=[]):
        CNN_layers=self.CNN_layers 
        CNN_heads=[1]*len(CNN_layers)
        """
        layer definition
        """
        start_time=time.time()
        assert len(CNN_layers)==len(CNN_heads)

        In=list()
        """
        #[#agents,batch,feature_dim],[#agents,batch,neighbors,agents],[batch,1,neighbors]
        ->[#agentsxbatch,feature_dim],[#agentsxbatch,neighbors,agents],[batch,1,neighbors]
        """
        #In: [batch,agent,feature]
        #In: [batch,agent,in_neighbors]
        #In: [batch,agent,neighbors,agents]
        In.append(Input(shape=[self.num_agents,self.len_feature],name="feature"))
        # w_in
        In.append(Input(shape=[self.num_agents,self.num_in_neighbors],name="w_in"))
        raw_feature_and_w_in = Concatenate(axis=-1)([In[0], In[1]])
        In.append(Input(shape=(self.num_agents,self.num_neighbors,self.num_agents),name="adjacency_matrix"))


        Input_end_time=time.time()
        """
        Currently, the MLP layer 
        -input: [batch,agent,feature_dim]
        -outpout: [#agent,batch,128]
        """
        feature=self.MLP(raw_feature_and_w_in,MLP_layers)

        Embedding_end_time=time.time()


        #TODO: remove the dense setting
        #feature:[batch,agents,feature_dim]
        att_record_all_layers=list()
        print("CNN_heads:", CNN_heads)
        for CNN_layer_index,CNN_layer_size in enumerate(CNN_layers):
            print("CNN_heads[CNN_layer_index]:",CNN_heads[CNN_layer_index])
            if CNN_layer_index==0:
                h,att_record=self.MultiHeadsAttModel(
                    feature,
                    In[2],
                    l=self.num_neighbors,
                    d=MLP_layers[-1],
                    dv=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    nv=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                    )
            else:
                h,att_record=self.MultiHeadsAttModel(
                    h,
                    In[2],
                    l=self.num_neighbors,
                    d=MLP_layers[-1],
                    dv=CNN_layer_size[0],
                    dout=CNN_layer_size[1],
                    nv=CNN_heads[CNN_layer_index],
                    suffix=CNN_layer_index
                    )
            att_record_all_layers.append(att_record)

        if len(CNN_layers)>1:
            att_record_all_layers=Concatenate(axis=1)(att_record_all_layers)
        else:
            att_record_all_layers=att_record_all_layers[0]

        att_record_all_layers=Reshape(
            (len(CNN_layers),self.num_agents,CNN_heads[-1],self.num_neighbors)
            )(att_record_all_layers)

        
        #TODO remove dense net
        for layer_index,layer_size in enumerate(Output_layers):
                h=Dense(layer_size,activation='relu',kernel_initializer='random_normal',name='Dense_q_%d'%layer_index)(h)
        #action prediction layer
        #[batch,agent,32]->[batch,agent,action]
        out = Dense(self.num_actions,kernel_initializer='random_normal',name='action_layer',
            activity_regularizer=L2ReLURegularizer(l=1e-1))(h)
        #out:[batch,agent,action], att:[batch,layers,agent,head,neighbors]
        model=Model(inputs=In,outputs=[out,att_record_all_layers])

        if self.att_regulatization:
            model.compile(
                optimizer=RMSprop(lr=self.lr),
                loss=[self.dic_agent_conf["LOSS_FUNCTION"],'kullback_leibler_divergence'],
                loss_weights=[1,self.dic_agent_conf["rularization_rate"]])
        else:
            model.compile(
                optimizer=RMSprop(lr=self.lr),
                loss=self.dic_agent_conf["LOSS_FUNCTION"],
                loss_weights=[1,0])
        # model.compile(optimizer=Adam(lr = 0.0001), loss='mse')
        model.summary()
        network_end=time.time()
        print('build_Input_end_time：',Input_end_time-start_time)
        print('embedding_time:',Embedding_end_time-Input_end_time)
        print('total time:',network_end-start_time)
        return model

    # w_in
    def build_v_network(self, bar=False):

        In=list()
        """
        #[#agents,batch,feature_dim],[#agents,batch,neighbors,agents],[batch,1,neighbors]
        ->[#agentsxbatch,feature_dim],[#agentsxbatch,neighbors,agents],[batch,1,neighbors]
        """
        #In: [batch,agent,feature]
        #In: [batch,agent,in_neighbors]
        #In: [batch,agent,neighbors,agents]
        In.append(Input(shape=[self.num_agents,self.len_feature],name="feature"))
        In.append(Input(shape=[self.num_agents,self.num_in_neighbors],name="w_in"))
        In.append(Input(shape=(self.num_agents,self.num_neighbors,self.num_agents),name="adjacency_matrix"))

        # reuse
        if not bar:
            q, _ = self.q_network(In)
        else:
            q, _ = self.q_network_bar(In)
        v = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(q)

        network = Model(inputs=In, outputs=v)

        return network

    def v_gradient(self):
        """get critic gradient function.

        Returns:
            function, gradient function for critic.
        """
        #In: [batch,agent,feature]
        #In: [batch,agent,in_neighbors]
        #In: [batch,agent,neighbors,agents]
        #Out: [batch,agent,1]
        vinput = self.v_network.input
        voutput = self.v_network.output

        # compute the gradient of the action with q value, dq/da.
        w_in_grads = K.gradients(voutput, vinput[1])[0]
        # w_in to w_out
        full_w_grad_matrix = [[None for j in range(self.num_agents)]
            for i in range(self.num_agents)]

        # -1 means to take all samples in batch dimension
        empty_for_padding = tf.slice(w_in_grads, [0,0,0], [-1,1,1])
        for inter_ind, inter in enumerate(self.inter_info):
            in_adjacency_row = inter.in_adjacency_row
            in_neighbors = in_adjacency_row[in_adjacency_row != -1]
            for j, neighbor_inter_ind in enumerate(in_neighbors):
                # -1 means to take all samples in batch dimension
                full_w_grad_matrix[inter_ind][neighbor_inter_ind] = tf.slice(w_in_grads, [0,inter_ind,j], [-1,1,1])
            # illegal places remain None
                
        full_w_grad_matrix = list(map(list, zip(*full_w_grad_matrix))) # transpose
        full_w_out_grads = [[full_w_grad_matrix[i][j] for j in range(self.num_agents) if full_w_grad_matrix[i][j] is not None]
            for i in range(self.num_agents)]
        # padding empty_for_padding
        max_len = self.num_neighbors
        total_w_out_grads = []
        for i in range(self.num_agents):
            # padding
            padding_w_out_grads = full_w_out_grads[i] + [empty_for_padding for _ in range(max_len-len(full_w_out_grads[i]))]
            total_w_out_grads.append(tf.concat(padding_w_out_grads, axis=2))
        w_out_grads = tf.concat(total_w_out_grads, axis=1)

        return K.function(vinput, [w_out_grads])

    def w_optimizer(self):
        """w_optimizer.

        Returns:
            function, opt function for w actor.
        """
        self.ainput = self.w_network.input
        aoutput_before, _ = self.w_network.output
        aoutput = Activation('softmax')(aoutput_before)
        trainable_weights = self.w_network.trainable_weights
        self.action_gradient = tf.placeholder(tf.float32, shape=(None, self.num_agents, self.num_neighbors))

        # tf.gradients will calculate dy/dx with a initial gradients for y
        # action_gradient is dq / da, so this is dq/da * da/dparams
        # negative, for maximizing global V
        params_grad = tf.gradients(aoutput, trainable_weights, -self.action_gradient)
        grads = zip(params_grad, trainable_weights)
        # w learning rate is still positive: for AdamOptimizer
        self.opt = tf.train.AdamOptimizer(self.w_lr).apply_gradients(grads)
        # self.sess.run(tf.global_variables_initializer())

    def train_w_network(self, dic_exp_conf):

        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            epochs = 1000
        else:
            epochs = self.w_epochs
        # batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))
        batch_size = self.dic_agent_conf["BATCH_SIZE"]

        # myTODO: EarlyStopping, validation_split
        # early_stopping = EarlyStopping(
        #     monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')

        # # hist = self.q_network.fit(self.Xs, self.Y, batch_size=batch_size, epochs=epochs,
        # #                           shuffle=False,
        # #                           verbose=2, validation_split=0.3, callbacks=[early_stopping])
        # hist = self.q_network.fit(self.Xs, self.Y_total, batch_size=batch_size, epochs=epochs,
        #                           shuffle=False,
        #                           verbose=2, validation_split=0.3,
        #                           callbacks=[early_stopping,TensorBoard(log_dir='./temp.tensorboard')])
        # softmax
        def softmax(x):
            shifted_x = x - np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(shifted_x)
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        _features, _, _adjs = self.Xs
        w_Xs = [_features,_adjs]
        for epoch in range(epochs):
            print('Epoch %d/%d' % (epoch+1, epochs)) # verbose=2
            _new_w_out_before, _ = self.w_network.predict(w_Xs)
            _new_w_out = softmax(_new_w_out_before)
            # use adjs: batch_size, num_agents, num_neighbors, num_agents
            _new_w_in = self.w_out_to_w_in(_new_w_out, raw_total_adjs=None, adjs=_adjs)

            q_values, _ = self.q_network.predict([_features, _new_w_in, _adjs])
            v_values = np.max(q_values, axis=-1)
            global_v_values = np.sum(v_values, axis=-1)
            print('- v_value: ', np.mean(global_v_values))
            
            new_w_out_grads = np.array(self.get_v_grad([_features, _new_w_in, _adjs])[0])
            # mask
            new_w_out_grads *= self.v_grad_mask

            num_samples = _features.shape[0]
            num_complete_batches = num_samples // batch_size # number of batches of size batch_size in your partitionning 
            for k in range(0, num_complete_batches):
                my_feed_dict = {
                    i: d[k*batch_size:(k+1)*batch_size] for i, d in zip(self.ainput, w_Xs)
                }
                my_feed_dict[self.action_gradient] = new_w_out_grads[k*batch_size:(k+1)*batch_size] / batch_size
                _ = self.sess.run(self.opt, feed_dict=my_feed_dict)
            # Handling the end case (last batch < batch_size) 
            if num_samples % batch_size != 0:
                my_feed_dict = {
                    i: d[num_complete_batches*batch_size:] for i, d in zip(self.ainput, w_Xs)
                }
                my_feed_dict[self.action_gradient] = new_w_out_grads[num_complete_batches*batch_size:] / (num_samples % batch_size)
                _ = self.sess.run(self.opt, feed_dict=my_feed_dict)
        

    def load_w_network(self, file_name, file_path=None, w_out_mask=None, self_mask=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        if w_out_mask is None or self_mask is None:
            self.w_network = load_model(
                os.path.join(file_path, "%s.h5" % file_name),
                custom_objects={'RepeatVector3D':RepeatVector3D, 'L2ReLURegularizer':L2ReLURegularizer})
        else:
            self.w_network = self.build_w_network(w_out_mask, self_mask)
            self.w_network.load_weights(os.path.join(file_path, "%s.h5" % file_name))
        print("succeed in loading model %s"%file_name)

    def load_w_network_bar(self, file_name, file_path=None, w_out_mask=None, self_mask=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        if w_out_mask is None or self_mask is None:
            self.w_network_bar = load_model(
                os.path.join(file_path, "%s.h5" % file_name),
                custom_objects={'RepeatVector3D':RepeatVector3D, 'L2ReLURegularizer':L2ReLURegularizer})
        else:
            self.w_network_bar = self.build_w_network(w_out_mask, self_mask)
            self.w_network_bar.load_weights(os.path.join(file_path, "%s.h5" % file_name))
        print("succeed in loading model %s"%file_name) 

    def save_w_network(self, file_name, whole_model=False):
        if whole_model:
            self.w_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))
        else:
            self.w_network.save_weights(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def save_w_network_bar(self, file_name, whole_model=False):
        # --- my modification ---
        actor_weights = self.w_network.get_weights()
        actor_target_weights = self.w_network_bar.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.dic_agent_conf["TAU"] * actor_weights[i] + (1-self.dic_agent_conf["TAU"])* actor_target_weights[i]
        self.w_network_bar.set_weights(actor_target_weights)

        if whole_model:
            self.w_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))
        else:
            self.w_network_bar.save_weights(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))
