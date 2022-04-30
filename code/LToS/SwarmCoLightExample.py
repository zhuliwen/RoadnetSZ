import os
import math
import copy

import numpy as np
import pandas as pd
import pickle as pkl

import config
import runexp
import model_test

class SwarmCoLightExample(object):
    """docstring for SwarmCoLightExample"""

    def __init__(self, model_path, model_round):
        '''
        model_path: model base path
        model round: round after model path
        Without any new training process, please just let the following arguments in function main be.
        '''
        super(SwarmCoLightExample, self).__init__()
        
        # load configurations
        self.model_path = model_path
        self.model_round = model_round
        '''
        args = runexp.parse_args()
        #memo = "multi_phase/optimal_search_new/new_headway_anon"

        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
        self.set_dic_traffic_env_conf(args.memo, args.env, args.road_net, args.gui, args.volume,
            args.suffix, args.mod, args.cnt, args.gen, args.all, args.workers,
            args.onemodel)
        '''
        visible_gpu = '-1'
        memo = '0515_afternoon_Colight_6_6_bi'
        env = 1
        road_net = '6_6'
        gui = False
        volume = '300'
        suffix = '0.3_bi'
        mod = 'SwarmCoLight'
        cnt = 3600
        gen = 4
        _all = False
        workers = 7
        onemodel = False
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpu
        self.set_dic_traffic_env_conf(memo, env, road_net, gui, volume,
            suffix, mod, cnt, gen, _all, workers,
            onemodel)

    def set_dic_traffic_env_conf(self, memo, env, road_net, gui, volume, suffix, mod, cnt, gen, r_all, workers, onemodel):
        # main(args.memo, args.env, args.road_net, args.gui, args.volume, args.ratio, args.mod, args.cnt, args.gen)
        #Jinan_3_4
        NUM_COL = int(road_net.split('_')[0])
        NUM_ROW = int(road_net.split('_')[1])
        num_intersections = NUM_ROW * NUM_COL
        # print('num_intersections:',num_intersections)

        ENVIRONMENT = ["sumo", "anon"][env]

        # if r_all:
        if False:
            traffic_file_list = [ENVIRONMENT+"_"+road_net+"_%d_%s" %(v,suffix) for v in range(100,400,100)]
        else:
            traffic_file="{0}_{1}_{2}_{3}".format(ENVIRONMENT, road_net, volume, suffix)

        # if env:
        if True:
            traffic_file += ".json"
        else:
            traffic_file_list = [i+ ".xml" for i in traffic_file_list ]

        process_list = []
        n_workers = workers     #len(traffic_file_list)

        # multi_process = True
        multi_process = False
        TOP_K_ADJACENCY=5
        TOP_K_ADJACENCY_LANE=5
        PRETRAIN=False
        NUM_ROUNDS=100
        EARLY_STOP=False
        NEIGHBOR=False
        SAVEREPLAY=False
        ADJACENCY_BY_CONNECTION_OR_GEO=True
        hangzhou_archive=False
        ANON_PHASE_REPRE=[]

        if 'CoLight_Signal' in mod:
            #12dim
            ANON_PHASE_REPRE={
                # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],# 'WSES',
                2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],# 'NSSS',
                3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],# 'WLEL',
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]# 'NLSL',
            }
        else:
            #12dim
            ANON_PHASE_REPRE={
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0]
            }

        # global PRETRAIN
        # global NUM_ROUNDS
        # global EARLY_STOP

        # global TOP_K_ADJACENCY
        # global TOP_K_ADJACENCY_LANE
        # global NEIGHBOR
        # global SAVEREPLAY
        # global ADJACENCY_BY_CONNECTION_OR_GEO
        # global ANON_PHASE_REPRE 

        dic_traffic_env_conf_extra = {
            "USE_LANE_ADJACENCY": True,
            "ONE_MODEL": onemodel,
            "NUM_AGENTS": num_intersections,
            "NUM_INTERSECTIONS": num_intersections,
            "ACTION_PATTERN": "set",
            "MEASURE_TIME": 10,
            "IF_GUI": gui,
            "DEBUG": False,
            "TOP_K_ADJACENCY": TOP_K_ADJACENCY,
            "ADJACENCY_BY_CONNECTION_OR_GEO": ADJACENCY_BY_CONNECTION_OR_GEO,
            "TOP_K_ADJACENCY_LANE": TOP_K_ADJACENCY_LANE,
            "SIMULATOR_TYPE": ENVIRONMENT,
            "BINARY_PHASE_EXPANSION": True,
            "FAST_COMPUTE": True,

            "NEIGHBOR": NEIGHBOR,
            "MODEL_NAME": mod,



            "SAVEREPLAY": SAVEREPLAY,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,

            "TRAFFIC_FILE": traffic_file,
            "VOLUME": volume,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            "phase_expansion": {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0],
                5: [1, 1, 0, 0, 0, 0, 0, 0],
                6: [0, 0, 1, 1, 0, 0, 0, 0],
                7: [0, 0, 0, 0, 0, 0, 1, 1],
                8: [0, 0, 0, 0, 1, 1, 0, 0]
            },

            "phase_expansion_4_lane": {
                1: [1, 1, 0, 0],
                2: [0, 0, 1, 1],
            },


            "LIST_STATE_FEATURE": [
                "cur_phase",
                # "time_this_phase",
                # "vehicle_position_img",
                # "vehicle_speed_img",
                # "vehicle_acceleration_img",
                # "vehicle_waiting_time_img",
                "lane_num_vehicle",
                # "lane_num_vehicle_been_stopped_thres01",
                # "lane_num_vehicle_been_stopped_thres1",
                # "lane_queue_length",
                # "lane_num_vehicle_left",
                # "lane_sum_duration_vehicle_left",
                # "lane_sum_waiting_time",
                # "terminal",
                # "coming_vehicle",
                # "leaving_vehicle",
                # "pressure"

                # "adjacency_matrix",
                # "lane_queue_length",
                # "connectivity",

                # adjacency_matrix_lane
            ],

                "DIC_FEATURE_DIM": dict(
                    D_LANE_QUEUE_LENGTH=(4,),
                    D_LANE_NUM_VEHICLE=(4,),

                    D_COMING_VEHICLE = (12,),
                    D_LEAVING_VEHICLE = (12,),

                    D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
                    D_CUR_PHASE=(1,),
                    D_NEXT_PHASE=(1,),
                    D_TIME_THIS_PHASE=(1,),
                    D_TERMINAL=(1,),
                    D_LANE_SUM_WAITING_TIME=(4,),
                    D_VEHICLE_POSITION_IMG=(4, 60,),
                    D_VEHICLE_SPEED_IMG=(4, 60,),
                    D_VEHICLE_WAITING_TIME_IMG=(4, 60,),

                    D_PRESSURE=(1,),

                    D_ADJACENCY_MATRIX=(2,),

                    D_ADJACENCY_MATRIX_LANE=(6,),

                ),

            "DIC_REWARD_INFO": {
                "flickering": 0,#-5,#
                "sum_lane_queue_length": 0,
                "sum_lane_wait_time": 0,
                "sum_lane_num_vehicle_left": 0,#-1,#
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": -0.25,
                "pressure": 0  # -0.25
            },

            "LANE_NUM": {
                "LEFT": 1,
                "RIGHT": 1,
                "STRAIGHT": 1
            },

            "PHASE": {
                "sumo": {
                    0: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                    1: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                    2: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                    3: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                },

                # "anon": {
                #     # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                #     1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],# 'WSES',
                #     2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],# 'NSSS',
                #     3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],# 'WLEL',
                #     4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]# 'NLSL',
                #     # 'WSWL',
                #     # 'ESEL',
                #     # 'WSES',
                #     # 'NSSS',
                #     # 'NSNL',
                #     # 'SSSL',
                # },
                "anon":ANON_PHASE_REPRE,
                # "anon": {
                #     # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                #     1: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                #     2: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                #     3: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                #     4: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                #     # 'WSWL',
                #     # 'ESEL',
                #     # 'WSES',
                #     # 'NSSS',
                #     # 'NSNL',
                #     # 'SSSL',
                # },
            }
        }

        ## ==================== multi_phase ====================
        # global hangzhou_archive
        if hangzhou_archive:
            template='Archive+2'
        elif volume=='jinan':
            template="Jinan"
        elif volume=='hangzhou':
            template='Hangzhou'
        elif volume=='newyork':
            template='NewYork'
        elif volume=='chacha':
            template='Chacha'
        elif volume=='dynamic_attention':
            template='dynamic_attention'
        elif dic_traffic_env_conf_extra["LANE_NUM"] == config._LS:
            template = "template_ls"
        elif dic_traffic_env_conf_extra["LANE_NUM"] == config._S:
            template = "template_s"
        elif dic_traffic_env_conf_extra["LANE_NUM"] == config._LSR:
            # template = "template_lsr"
            template = "small"
        else:
            raise ValueError

        if dic_traffic_env_conf_extra['NEIGHBOR']:
            list_feature = dic_traffic_env_conf_extra["LIST_STATE_FEATURE"].copy()
            for feature in list_feature:
                for i in range(4):
                    dic_traffic_env_conf_extra["LIST_STATE_FEATURE"].append(feature+"_"+str(i))

        if mod in ['SwarmCoLight','CoLight','GCN','SimpleDQNOne']:
            dic_traffic_env_conf_extra["NUM_AGENTS"] = 1
            dic_traffic_env_conf_extra['ONE_MODEL'] = False
            if "adjacency_matrix" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE'] and \
                "adjacency_matrix_lane" not in dic_traffic_env_conf_extra['LIST_STATE_FEATURE'] and \
                mod not in ['SimpleDQNOne']:
                dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("adjacency_matrix")
                dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("adjacency_matrix_lane")
                if dic_traffic_env_conf_extra['ADJACENCY_BY_CONNECTION_OR_GEO']:
                    TOP_K_ADJACENCY = 5
                    dic_traffic_env_conf_extra['LIST_STATE_FEATURE'].append("connectivity")
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CONNECTIVITY'] = \
                        (5,)
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX'] = \
                        (5,)
                else:
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX'] = \
                        (dic_traffic_env_conf_extra['TOP_K_ADJACENCY'],)

                if dic_traffic_env_conf_extra['USE_LANE_ADJACENCY']:
                    dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_ADJACENCY_MATRIX_LANE'] = \
                        (dic_traffic_env_conf_extra['TOP_K_ADJACENCY_LANE'],)
        else:
            dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]

        if dic_traffic_env_conf_extra['BINARY_PHASE_EXPANSION']:
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE'] = (8,)
            if dic_traffic_env_conf_extra['NEIGHBOR']:
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)
            else:

                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)

        def merge(dic_tmp, dic_to_change):
            dic_result = copy.deepcopy(dic_tmp)
            dic_result.update(dic_to_change)

            return dic_result
        self.dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)

    def __call__(self, run_counts):
        '''
        Input: episode length; other arguments are offered in the form of files.
        Output: pandas.dataframe which contains the following performance: {average duration (travel time); queue length (average waiting queue length); vehicle_in (number of vehicles that managed to come in), vehicle_out (number of vehicles that managed to go out)};
        Large amount of other statistics will also be provided to stdout and in the folder records/ and summary/, and thus you'd better run with nohup command.
		'''
        model_test.test(self.model_path, self.model_round, run_counts, self.dic_traffic_env_conf, if_gui=False)
        return self.summary_detail_single_test(run_counts)

    def summary_detail_single_test(self, run_counts):
        time_interval = 120
        num_seg = math.ceil(run_counts/time_interval)

        round_dir = os.path.join(self.model_path.replace("model", "records"), "test_round", "round_%d" % self.model_round)
        # print("===={0}".format(round))
        round_summary = {}

        nan_thres = 120
        df_vehicle_all = []
        duration_each_round_list = []
        duration_each_round_list2 = []
        queue_length_each_round_list = []
        min_queue_length = min_duration = min_duration2 = float('inf')
        min_queue_length_id = min_duration_ind = 0
        
        queue_length_each_round = []
        num_of_vehicle_in = []
        num_of_vehicle_out = []

        list_duration_seg = [float('inf')] * num_seg
        list_queue_length_seg = [float('inf')] * num_seg
        list_queue_length_id_seg = [0] * num_seg
        list_duration_id_seg = [0] * num_seg
        NAN_LABEL = -1
        num_intersection = self.dic_traffic_env_conf['NUM_INTERSECTIONS']
        for inter_index in range(num_intersection):
            try:
                # summary items (queue_length) from pickle
                f = open(os.path.join(round_dir, "inter_{0}.pkl".format(inter_index)), "rb")
                samples = pkl.load(f)
                queue_length_each_inter_each_round = 0
                for sample in samples:
                    queue_length_each_inter_each_round += sum(sample['state']['lane_num_vehicle_been_stopped_thres1'])
                queue_length_each_inter_each_round = queue_length_each_inter_each_round//len(samples)
                f.close()

                # summary items (duration) from csv
                df_vehicle_inter = pd.read_csv(os.path.join(round_dir, "vehicle_inter_{0}.csv".format(inter_index)),
                                                 sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                 names=["vehicle_id", "enter_time", "leave_time"])
                df_vehicle_inter['leave_time_origin'] = df_vehicle_inter['leave_time']
                df_vehicle_inter['leave_time'].fillna(run_counts,inplace=True)
                df_vehicle_inter['duration'] = df_vehicle_inter["leave_time"].values - df_vehicle_inter["enter_time"].values
                ave_duration = df_vehicle_inter['duration'].mean(skipna=True)
                # print("------------- inter_index: {0}\tave_duration: {1}\tave_queue_length:{2}"
                #       .format(inter_index, ave_duration, queue_length_each_inter_each_round))

                # if "peak" in traffic_file:
                #     did1 = df_vehicle_inter_0["enter_time"].values <= run_counts / 2
                #     duration = df_vehicle_inter_0["leave_time"][did1].values - df_vehicle_inter_0["enter_time"][
                #         did1].values
                #     ave_duration = np.mean([time for time in duration if not isnan(time)])
                #
                #     did2 = df_vehicle_inter_0["enter_time"].values > run_counts / 2
                #     duration2 = df_vehicle_inter_0["leave_time"][did2].values - df_vehicle_inter_0["enter_time"][
                #         did2].values
                #     ave_duration2 = np.mean([time for time in duration2 if not isnan(time)])
                #     duration_each_round_list2.append(ave_duration2)
                #
                #     real_traffic_vol2 = 0       
                #     nan_num2 = 0
                #     for time in duration2:
                #         if not isnan(time):
                #             real_traffic_vol2 += 1
                #         else:
                #             nan_num2 += 1
                #
                #     if nan_num2 < nan_thres:
                #         if min_duration2 > ave_duration2 and ave_duration2 > 24:
                #             min_duration2 = ave_duration2
                #             min_duration_ind2 = int(round[6:])                                                                        
                df_vehicle_all.append(df_vehicle_inter)
                queue_length_each_round.append(queue_length_each_inter_each_round)      

            except:
                queue_length_each_round.append(NAN_LABEL)
                # num_of_vehicle_in.append(NAN_LABEL)
                # num_of_vehicle_out.append(NAN_LABEL)     
        # if len(df_vehicle_all)==0:
        #     print("====================================EMPTY")
        #     continue

        df_vehicle_all = pd.concat(df_vehicle_all)
        # vehicle_duration = df_vehicle_all.groupby(by=['vehicle_id'])['duration'].sum()
        # ave_duration = vehicle_duration.mean()
        # --- my modification ---
        # vehicle_duration = []
        # for vehicle_id, group in df_vehicle_all.sort_values(by='enter_time').groupby(by='vehicle_id'):
        #     true_duration = 0
        #     beg, end = -1, -1
        #     for i, row in group.iterrows():
        #         print(vehicle_id, row['enter_time'], row['leave_time'])
        #         if row['enter_time'] < end:
        #             end = max(end, row['leave_time'])
        #         else:
        #             true_duration += end - beg
        #             beg, end = row['enter_time'], row['leave_time']
        #     true_duration += end - beg
        #     vehicle_duration.append(true_duration)
        df_vehicle_all_group = df_vehicle_all.groupby(by=['vehicle_id'])
        vehicle_duration = pd.DataFrame((df_vehicle_all_group['leave_time'].max() - df_vehicle_all_group['enter_time'].min()),
            columns=['duration'])
        ave_duration = vehicle_duration['duration'].mean()

        ave_queue_length = np.mean(queue_length_each_round)

        duration_each_round_list.append(ave_duration)
        queue_length_each_round_list.append(ave_queue_length)


        num_of_vehicle_in.append(len(df_vehicle_all['vehicle_id'].unique()))
        num_of_vehicle_out.append(len(df_vehicle_all.dropna()['vehicle_id'].unique()))

        # print("==== round: {0}\tave_duration: {1}\tave_queue_length_per_intersection:{2}\t"
        #       "num_of_vehicle_in:{3}\tnum_of_vehicle_out:{4}"
        #       .format(round, ave_duration,ave_queue_length,num_of_vehicle_in[-1],num_of_vehicle_out[-1]))

        duration_flow = vehicle_duration.reset_index()

        duration_flow['direction'] = duration_flow['vehicle_id'].apply(lambda x:x.split('_')[1])
        duration_flow_ave = duration_flow.groupby(by=['direction'])['duration'].mean()
        print(duration_flow_ave)

        # print(real_traffic_vol, traffic_vol, traffic_vol - real_traffic_vol, nan_num)
        if min_queue_length > ave_queue_length:
            min_queue_length = np.mean(queue_length_each_round)
            # min_queue_length_id = int(round[6:])
            min_queue_length_id = self.model_round
        #
        # valid_flag = json.load(open(os.path.join(round_dir, "valid_flag.json")))
        # if valid_flag['0']:  # temporary for one intersection
        #     nan_num2 = 0
        #     if min_duration > ave_duration and ave_duration > 24:
        #         min_duration = ave_duration
        #         min_duration_ind = int(round[6:])


        #### This is for long time

        if num_seg > 1:
            for i, interval in enumerate(range(0, run_counts, time_interval)):
                did = df_vehicle_all[(df_vehicle_all["enter_time"]< interval+time_interval) &
                                     (df_vehicle_all["enter_time"].values > interval)]
                #vehicle_in_seg = sum([int(x) for x in (df_vehicle_inter_0["enter_time"][did].values > 0)])
                #vehicle_out_seg = sum([int(x) for x in (df_vehicle_inter_0["leave_time"][did].values > 0)])

                vehicle_duration_seg = did.groupby(by=['vehicle_id'])['duration'].sum()
                ave_duration_seg = vehicle_duration_seg[vehicle_duration_seg>10].mean()
                # print(traffic_file, round, i, ave_duration)
                # real_traffic_vol_seg = 0
                # nan_num_seg = 0
                # for time in duration_seg:
                #     if not isnan(time):
                #         real_traffic_vol_seg += 1
                #     else:
                #         nan_num_seg += 1

                # print(real_traffic_vol, traffic_vol, traffic_vol - real_traffic_vol, nan_num)
                nan_num_seg = did['leave_time_origin'].isna().sum()

                if nan_num_seg < nan_thres:
                    list_duration_seg[i] = ave_duration_seg
                    # list_duration_id_seg[i] = int(round[6:])
                    list_duration_id_seg[i] = self.model_round

                #round_summary = {}
                for j in range(num_seg):
                    key = "min_duration-" + str(j)

                    if key not in round_summary.keys():
                        round_summary[key] = [list_duration_seg[j]]
                    else:
                        round_summary[key].append(list_duration_seg[j])
                #round_result_dir = os.path.join("summary", memo, traffic_file)
                #if not os.path.exists(round_result_dir):
                #    os.makedirs(round_result_dir)

        result_dir = self.model_path.replace("model", "records")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        _res = {
            "duration": duration_each_round_list,
            "queue_length": queue_length_each_round_list,
            "vehicle_in": num_of_vehicle_in,
            "vehicle_out": num_of_vehicle_out
        }
        result = pd.DataFrame(_res)
        result.to_csv(os.path.join(result_dir, "test_results.csv"))
        return result



if __name__=="__main__":
    sw = SwarmCoLightExample(
        model_path='model/0515_afternoon_Colight_6_6_bi/anon_6_6_300_0.3_bi.json_05_21_22_24_02',
        model_round=94)

    result = sw(run_counts=3600)
    print('result: ', result)
