''''
No 0
roadnet_3_3.json
anon_3_3_300_0.3_bi1_raw.json
'''
dic_traffic_env_conf_3_3_raw = {
        "ADJACENCY_BY_CONNECTION_OR_GEO": True,
        "USE_LANE_ADJACENCY": True,
        "TRAFFIC_FILE": "anon_3_3_300_0.3_bi1_raw.json",
        "THREADNUM": 8,
        "SAVEREPLAY": False,
        "RLTRAFFICLIGHT": True,
        "INTERVAL": 1,
        "NUM_INTERSECTIONS": 9,  # attention
        "ACTION_PATTERN": "switch",  # set
        "MEASURE_TIME": 10,
        "MIN_ACTION_TIME": 5,  # 10
        "YELLOW_TIME": 5,
        "ALL_RED_TIME": 0,
        "DEBUG": False,  # False
        "BINARY_PHASE_EXPANSION": True,
        "FAST_COMPUTE": True,
        'NUM_AGENTS': 1,

        "NEIGHBOR": False,
        "MODEL_NAME": "STGAT",
        "SIMULATOR_TYPE": "anon",
        "TOP_K_ADJACENCY": 9,
        "TOP_K_ADJACENCY_LANE": 6,

        "SAVEREPLAY": False,
        "NUM_ROW": 3,  # attention
        "NUM_COL": 3,  # attention

        "VOLUME": 300,
        "ROADNET_FILE": "roadnet_{0}.json".format("3_3"),  # attention

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
            # "adjacency_matrix_lane",
        ],

        "DIC_FEATURE_DIM": dict(
            D_LANE_QUEUE_LENGTH=(4,),
            D_LANE_NUM_VEHICLE=(4,),

            D_COMING_VEHICLE=(12,),
            D_LEAVING_VEHICLE=(12,),

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
            D_ADJACENCY_MATRIX_LANE=(6,)

        ),

        "DIC_REWARD_INFO": {
            "flickering": 0,
            "sum_lane_queue_length": 0,
            "sum_lane_wait_time": 0,
            "sum_lane_num_vehicle_left": 0,
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
                0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
                1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
                2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
                3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
            },
            "anon": {
                # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
                2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
                3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
                # 'WSWL',
                # 'ESEL',
                # 'WSES',
                # 'NSSS',
                # 'NSNL',
                # 'SSSL',
            },
        }
    }
infos_3_3 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False},
                 {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False},
                 {'task': [5, 5], 'done_mdp': False}, {'task': [6, 6], 'done_mdp': False},
                 {'task': [7, 7], 'done_mdp': False}, {'task': [8, 8], 'done_mdp': False},
                 {'task': [9, 9], 'done_mdp': False})


'''
No 1
roadnet_6_6.json
anon_6_6_300_0.3_bi_raw.json
'''
dic_traffic_env_conf_6_6_raw = {
        "ADJACENCY_BY_CONNECTION_OR_GEO": True,
        "USE_LANE_ADJACENCY": True,
        "TRAFFIC_FILE": "anon_6_6_300_0.3_bi_raw.json",
        "THREADNUM": 8,
        "SAVEREPLAY": False,
        "RLTRAFFICLIGHT": True,
        "INTERVAL": 1,
        "NUM_INTERSECTIONS": 36,  # attention
        "ACTION_PATTERN": "switch",  # set
        "MEASURE_TIME": 10,
        "MIN_ACTION_TIME": 5,  # 10
        "YELLOW_TIME": 5,
        "ALL_RED_TIME": 0,
        "DEBUG": False,  # False
        "BINARY_PHASE_EXPANSION": True,
        "FAST_COMPUTE": True,
        'NUM_AGENTS': 1,

        "NEIGHBOR": False,
        "MODEL_NAME": "STGAT",
        "SIMULATOR_TYPE": "anon",
        "TOP_K_ADJACENCY": 9,
        "TOP_K_ADJACENCY_LANE": 6,

        "SAVEREPLAY": False,
        "NUM_ROW": 6,  # attention
        "NUM_COL": 6,  # attention

        "VOLUME": 300,
        "ROADNET_FILE": "roadnet_{0}.json".format("6_6"),  # attention

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
            # "adjacency_matrix_lane",
        ],

        "DIC_FEATURE_DIM": dict(
            D_LANE_QUEUE_LENGTH=(4,),
            D_LANE_NUM_VEHICLE=(4,),

            D_COMING_VEHICLE=(12,),
            D_LEAVING_VEHICLE=(12,),

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
            D_ADJACENCY_MATRIX_LANE=(6,)

        ),

        "DIC_REWARD_INFO": {
            "flickering": 0,
            "sum_lane_queue_length": 0,
            "sum_lane_wait_time": 0,
            "sum_lane_num_vehicle_left": 0,
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
                0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
                1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
                2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
                3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
            },
            "anon": {
                # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
                2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
                3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
                # 'WSWL',
                # 'ESEL',
                # 'WSES',
                # 'NSSS',
                # 'NSNL',
                # 'SSSL',
            },
        }
    }
infos_6_6 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False},
                 {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False},
                 {'task': [5, 5], 'done_mdp': False}, {'task': [6, 6], 'done_mdp': False},
                 {'task': [7, 7], 'done_mdp': False}, {'task': [8, 8], 'done_mdp': False},
                 {'task': [9, 9], 'done_mdp': False}, {'task': [10, 10], 'done_mdp': False},
                 {'task': [11, 11], 'done_mdp': False}, {'task': [12, 12], 'done_mdp': False},
                 {'task': [13, 13], 'done_mdp': False}, {'task': [14, 14], 'done_mdp': False},
                 {'task': [15, 15], 'done_mdp': False}, {'task': [16, 16], 'done_mdp': False},
                 {'task': [17, 17], 'done_mdp': False}, {'task': [18, 18], 'done_mdp': False},
                 {'task': [19, 19], 'done_mdp': False}, {'task': [20, 20], 'done_mdp': False},
                 {'task': [21, 21], 'done_mdp': False}, {'task': [22, 22], 'done_mdp': False},
                 {'task': [23, 23], 'done_mdp': False}, {'task': [24, 24], 'done_mdp': False},
                 {'task': [25, 25], 'done_mdp': False}, {'task': [26, 26], 'done_mdp': False},
                 {'task': [27, 27], 'done_mdp': False}, {'task': [28, 28], 'done_mdp': False},
                 {'task': [29, 29], 'done_mdp': False}, {'task': [30, 30], 'done_mdp': False},
                 {'task': [31, 31], 'done_mdp': False}, {'task': [32, 32], 'done_mdp': False},
                 {'task': [33, 33], 'done_mdp': False}, {'task': [34, 34], 'done_mdp': False},
                 {'task': [35, 35], 'done_mdp': False}, {'task': [36, 36], 'done_mdp': False})


'''
No 2
roadnet_10_10.json
anon_10_10_300_0.3_bi_raw.json
'''
dic_traffic_env_conf_10_10_raw = {
        "ADJACENCY_BY_CONNECTION_OR_GEO": True,
        "USE_LANE_ADJACENCY": True,
        "TRAFFIC_FILE": "anon_10_10_300_0.3_bi_raw.json",
        "THREADNUM": 8,
        "SAVEREPLAY": False,
        "RLTRAFFICLIGHT": True,
        "INTERVAL": 1,
        "NUM_INTERSECTIONS": 100,  # attention
        "ACTION_PATTERN": "switch",  # set
        "MEASURE_TIME": 10,
        "MIN_ACTION_TIME": 5,  # 10
        "YELLOW_TIME": 5,
        "ALL_RED_TIME": 0,
        "DEBUG": False,  # False
        "BINARY_PHASE_EXPANSION": True,
        "FAST_COMPUTE": True,
        'NUM_AGENTS': 1,

        "NEIGHBOR": False,
        "MODEL_NAME": "STGAT",
        "SIMULATOR_TYPE": "anon",
        "TOP_K_ADJACENCY": 9,
        "TOP_K_ADJACENCY_LANE": 6,

        "SAVEREPLAY": False,
        "NUM_ROW": 10,  # attention
        "NUM_COL": 10,  # attention

        "VOLUME": 300,
        "ROADNET_FILE": "roadnet_{0}.json".format("10_10"),  # attention

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
            # "adjacency_matrix_lane",
        ],

        "DIC_FEATURE_DIM": dict(
            D_LANE_QUEUE_LENGTH=(4,),
            D_LANE_NUM_VEHICLE=(4,),

            D_COMING_VEHICLE=(12,),
            D_LEAVING_VEHICLE=(12,),

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
            D_ADJACENCY_MATRIX_LANE=(6,)

        ),

        "DIC_REWARD_INFO": {
            "flickering": 0,
            "sum_lane_queue_length": 0,
            "sum_lane_wait_time": 0,
            "sum_lane_num_vehicle_left": 0,
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
                0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
                1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
                2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
                3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
            },
            "anon": {
                # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
                2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
                3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
                # 'WSWL',
                # 'ESEL',
                # 'WSES',
                # 'NSSS',
                # 'NSNL',
                # 'SSSL',
            },
        }
    }
infos_10_10 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False}, {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False}, {'task': [5, 5], 'done_mdp': False},{'task': [6, 6], 'done_mdp': False},{'task': [7, 7], 'done_mdp': False},{'task': [8, 8], 'done_mdp': False},{'task': [9, 9], 'done_mdp': False},{'task': [10, 10], 'done_mdp': False},
                 {'task': [11, 11], 'done_mdp': False}, {'task': [12, 12], 'done_mdp': False},{'task': [13, 13], 'done_mdp': False}, {'task': [14, 14], 'done_mdp': False},{'task': [15, 15], 'done_mdp': False}, {'task': [16, 16], 'done_mdp': False},{'task': [17, 17], 'done_mdp': False}, {'task': [18, 18], 'done_mdp': False},{'task': [19, 19], 'done_mdp': False}, {'task': [20, 20], 'done_mdp': False},
                 {'task': [21, 21], 'done_mdp': False}, {'task': [22, 22], 'done_mdp': False},{'task': [23, 23], 'done_mdp': False}, {'task': [24, 24], 'done_mdp': False},{'task': [25, 25], 'done_mdp': False}, {'task': [26, 26], 'done_mdp': False},{'task': [27, 27], 'done_mdp': False}, {'task': [28, 28], 'done_mdp': False},{'task': [29, 29], 'done_mdp': False}, {'task': [30, 30], 'done_mdp': False},
                 {'task': [31, 31], 'done_mdp': False}, {'task': [32, 32], 'done_mdp': False},{'task': [33, 33], 'done_mdp': False}, {'task': [34, 34], 'done_mdp': False},{'task': [35, 35], 'done_mdp': False}, {'task': [36, 36], 'done_mdp': False},{'task': [37, 37], 'done_mdp': False}, {'task': [38, 38], 'done_mdp': False},{'task': [39, 39], 'done_mdp': False}, {'task': [40, 40], 'done_mdp': False},
                 {'task': [41, 41], 'done_mdp': False}, {'task': [42, 42], 'done_mdp': False},{'task': [43, 43], 'done_mdp': False}, {'task': [44, 44], 'done_mdp': False},{'task': [45, 45], 'done_mdp': False}, {'task': [46, 46], 'done_mdp': False},{'task': [47, 47], 'done_mdp': False}, {'task': [48, 48], 'done_mdp': False},{'task': [49, 49], 'done_mdp': False}, {'task': [50, 50], 'done_mdp': False},
                 {'task': [51, 51], 'done_mdp': False}, {'task': [52, 52], 'done_mdp': False},{'task': [53, 53], 'done_mdp': False}, {'task': [54, 54], 'done_mdp': False},{'task': [55, 55], 'done_mdp': False}, {'task': [56, 56], 'done_mdp': False},{'task': [57, 57], 'done_mdp': False}, {'task': [58, 58], 'done_mdp': False},{'task': [59, 59], 'done_mdp': False}, {'task': [60, 60], 'done_mdp': False},
                 {'task': [61, 61], 'done_mdp': False}, {'task': [62, 62], 'done_mdp': False},{'task': [63, 63], 'done_mdp': False}, {'task': [64, 64], 'done_mdp': False},{'task': [65, 65], 'done_mdp': False}, {'task': [66, 66], 'done_mdp': False},{'task': [67, 67], 'done_mdp': False}, {'task': [68, 68], 'done_mdp': False},{'task': [69, 69], 'done_mdp': False}, {'task': [70, 70], 'done_mdp': False},
                 {'task': [71, 71], 'done_mdp': False}, {'task': [72, 72], 'done_mdp': False},{'task': [73, 73], 'done_mdp': False}, {'task': [74, 74], 'done_mdp': False},{'task': [75, 75], 'done_mdp': False}, {'task': [76, 76], 'done_mdp': False},{'task': [77, 77], 'done_mdp': False}, {'task': [78, 78], 'done_mdp': False},{'task': [79, 79], 'done_mdp': False}, {'task': [80, 80], 'done_mdp': False},
                 {'task': [81, 81], 'done_mdp': False}, {'task': [82, 82], 'done_mdp': False},{'task': [83, 83], 'done_mdp': False}, {'task': [84, 84], 'done_mdp': False},{'task': [85, 85], 'done_mdp': False}, {'task': [86, 86], 'done_mdp': False},{'task': [87, 87], 'done_mdp': False}, {'task': [88, 88], 'done_mdp': False},{'task': [89, 89], 'done_mdp': False}, {'task': [90, 90], 'done_mdp': False},
                 {'task': [91, 91], 'done_mdp': False}, {'task': [92, 92], 'done_mdp': False},{'task': [93, 93], 'done_mdp': False}, {'task': [94, 94], 'done_mdp': False},{'task': [95, 95], 'done_mdp': False}, {'task': [96, 96], 'done_mdp': False},{'task': [97, 97], 'done_mdp': False}, {'task': [98, 98], 'done_mdp': False},{'task': [99, 99], 'done_mdp': False}, {'task': [100, 100], 'done_mdp': False},)


'''
No 3
roadnet_3_4.json
anon_3_4_jinan_real_raw.json
'''
dic_traffic_env_conf_3_4_raw = {
        "ADJACENCY_BY_CONNECTION_OR_GEO": True,
        "USE_LANE_ADJACENCY": True,
        "TRAFFIC_FILE": "anon_3_4_jinan_real_raw.json",
        "THREADNUM": 8,
        "SAVEREPLAY": False,
        "RLTRAFFICLIGHT": True,
        "INTERVAL": 1,
        "NUM_INTERSECTIONS": 12,  # attention
        "ACTION_PATTERN": "switch",  # set
        "MEASURE_TIME": 10,
        "MIN_ACTION_TIME": 5,  # 10
        "YELLOW_TIME": 5,
        "ALL_RED_TIME": 0,
        "DEBUG": False,  # False
        "BINARY_PHASE_EXPANSION": True,
        "FAST_COMPUTE": True,
        'NUM_AGENTS': 1,

        "NEIGHBOR": False,
        "MODEL_NAME": "STGAT",
        "SIMULATOR_TYPE": "anon",
        "TOP_K_ADJACENCY": 9,
        "TOP_K_ADJACENCY_LANE": 6,

        "SAVEREPLAY": False,
        "NUM_ROW": 4,  # attention
        "NUM_COL": 3,  # attention

        "VOLUME": 300,
        "ROADNET_FILE": "roadnet_{0}.json".format("3_4"),  # attention

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
            # "adjacency_matrix_lane",
        ],

        "DIC_FEATURE_DIM": dict(
            D_LANE_QUEUE_LENGTH=(4,),
            D_LANE_NUM_VEHICLE=(4,),

            D_COMING_VEHICLE=(12,),
            D_LEAVING_VEHICLE=(12,),

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
            D_ADJACENCY_MATRIX_LANE=(6,)

        ),

        "DIC_REWARD_INFO": {
            "flickering": 0,
            "sum_lane_queue_length": 0,
            "sum_lane_wait_time": 0,
            "sum_lane_num_vehicle_left": 0,
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
                0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
                1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
                2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
                3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
            },
            "anon": {
                # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
                2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
                3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
                # 'WSWL',
                # 'ESEL',
                # 'WSES',
                # 'NSSS',
                # 'NSNL',
                # 'SSSL',
            },
        }
    }
infos_3_4 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False}, {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False}, {'task': [5, 5], 'done_mdp': False},{'task': [6, 6], 'done_mdp': False},{'task': [7, 7], 'done_mdp': False},{'task': [8, 8], 'done_mdp': False},{'task': [9, 9], 'done_mdp': False},{'task': [10, 10], 'done_mdp': False},
                 {'task': [11, 11], 'done_mdp': False}, {'task': [12, 12], 'done_mdp': False})

'''
No 4
roadnet_4_4.json
anon_4_4_hangzhou_real_raw.json
'''
dic_traffic_env_conf_4_4_raw = {
    "ADJACENCY_BY_CONNECTION_OR_GEO": True,
    "USE_LANE_ADJACENCY": True,
    "TRAFFIC_FILE": "anon_4_4_hangzhou_real_raw.json",  # anon_4_4_hangzhou_real.json anon_3_3_300_0.3_bi.json# attention
    "THREADNUM": 8,
    "SAVEREPLAY": False,
    "RLTRAFFICLIGHT": True,
    "INTERVAL": 1,
    "NUM_INTERSECTIONS": 16,  # attention
    "ACTION_PATTERN": "switch",  # set
    "MEASURE_TIME": 10,
    "MIN_ACTION_TIME": 5,  # 10
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "DEBUG": False,  # False
    "BINARY_PHASE_EXPANSION": True,
    "FAST_COMPUTE": True,
    'NUM_AGENTS': 1,

    "NEIGHBOR": False,
    "MODEL_NAME": "STGAT",
    "SIMULATOR_TYPE": "anon",
    "TOP_K_ADJACENCY": 9,
    "TOP_K_ADJACENCY_LANE": 6,

    "SAVEREPLAY": False,
    "NUM_ROW": 4,  # attention
    "NUM_COL": 4,  # attention

    "VOLUME": 300,
    "ROADNET_FILE": "roadnet_{0}.json".format("4_4"),  # attention

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
        # "adjacency_matrix_lane",
    ],

    "DIC_FEATURE_DIM": dict(
        D_LANE_QUEUE_LENGTH=(4,),
        D_LANE_NUM_VEHICLE=(4,),

        D_COMING_VEHICLE=(12,),
        D_LEAVING_VEHICLE=(12,),

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
        D_ADJACENCY_MATRIX_LANE=(6,)

    ),

    "DIC_REWARD_INFO": {
        "flickering": 0,
        "sum_lane_queue_length": 0,
        "sum_lane_wait_time": 0,
        "sum_lane_num_vehicle_left": 0,
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
            0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
            1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
            2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
            3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
        },
        "anon": {
            # 0: [0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
            2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
            3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
            4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
            # 'WSWL',
            # 'ESEL',
            # 'WSES',
            # 'NSSS',
            # 'NSNL',
            # 'SSSL',
        },
    }
}
infos_4_4 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False}, {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False}, {'task': [5, 5], 'done_mdp': False},{'task': [6, 6], 'done_mdp': False},{'task': [7, 7], 'done_mdp': False},{'task': [8, 8], 'done_mdp': False},{'task': [9, 9], 'done_mdp': False},{'task': [10, 10], 'done_mdp': False},
                 {'task': [11, 11], 'done_mdp': False}, {'task': [12, 12], 'done_mdp': False},{'task': [13, 13], 'done_mdp': False}, {'task': [14, 14], 'done_mdp': False},{'task': [15, 15], 'done_mdp': False}, {'task': [16, 16], 'done_mdp': False})


'''
No 5
roadnet_16_3.json
anon_16_3_newyork_real_raw.json
'''
dic_traffic_env_conf_16_3_raw = {
    "ADJACENCY_BY_CONNECTION_OR_GEO": True,
    "USE_LANE_ADJACENCY": True,
    "TRAFFIC_FILE": "anon_16_3_newyork_real_raw.json",  # anon_4_4_hangzhou_real.json anon_3_3_300_0.3_bi.json# attention
    "THREADNUM": 8,
    "SAVEREPLAY": False,
    "RLTRAFFICLIGHT": True,
    "INTERVAL": 1,
    "NUM_INTERSECTIONS": 48,  # attention
    "ACTION_PATTERN": "switch",  # set
    "MEASURE_TIME": 10,
    "MIN_ACTION_TIME": 5,  # 10
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "DEBUG": False,  # False
    "BINARY_PHASE_EXPANSION": True,
    "FAST_COMPUTE": True,
    'NUM_AGENTS': 1,

    "NEIGHBOR": False,
    "MODEL_NAME": "STGAT",
    "SIMULATOR_TYPE": "anon",
    "TOP_K_ADJACENCY": 9,
    "TOP_K_ADJACENCY_LANE": 6,

    "SAVEREPLAY": False,
    "NUM_ROW": 3,  # attention
    "NUM_COL": 16,  # attention

    "VOLUME": 300,
    "ROADNET_FILE": "roadnet_{0}.json".format("16_3"),  # attention

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
        # "adjacency_matrix_lane",
    ],

    "DIC_FEATURE_DIM": dict(
        D_LANE_QUEUE_LENGTH=(4,),
        D_LANE_NUM_VEHICLE=(4,),

        D_COMING_VEHICLE=(12,),
        D_LEAVING_VEHICLE=(12,),

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
        D_ADJACENCY_MATRIX_LANE=(6,)

    ),

    "DIC_REWARD_INFO": {
        "flickering": 0,
        "sum_lane_queue_length": 0,
        "sum_lane_wait_time": 0,
        "sum_lane_num_vehicle_left": 0,
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
            0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
            1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
            2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
            3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
        },
        "anon": {
            # 0: [0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
            2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
            3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
            4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
            # 'WSWL',
            # 'ESEL',
            # 'WSES',
            # 'NSSS',
            # 'NSNL',
            # 'SSSL',
        },
    }
}
infos_16_3 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False}, {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False}, {'task': [5, 5], 'done_mdp': False},{'task': [6, 6], 'done_mdp': False},{'task': [7, 7], 'done_mdp': False},{'task': [8, 8], 'done_mdp': False},{'task': [9, 9], 'done_mdp': False},{'task': [10, 10], 'done_mdp': False},
                 {'task': [11, 11], 'done_mdp': False}, {'task': [12, 12], 'done_mdp': False},{'task': [13, 13], 'done_mdp': False}, {'task': [14, 14], 'done_mdp': False},{'task': [15, 15], 'done_mdp': False}, {'task': [16, 16], 'done_mdp': False},{'task': [17, 17], 'done_mdp': False}, {'task': [18, 18], 'done_mdp': False},{'task': [19, 19], 'done_mdp': False}, {'task': [20, 20], 'done_mdp': False},
                 {'task': [21, 21], 'done_mdp': False}, {'task': [22, 22], 'done_mdp': False},{'task': [23, 23], 'done_mdp': False}, {'task': [24, 24], 'done_mdp': False},{'task': [25, 25], 'done_mdp': False}, {'task': [26, 26], 'done_mdp': False},{'task': [27, 27], 'done_mdp': False}, {'task': [28, 28], 'done_mdp': False},{'task': [29, 29], 'done_mdp': False}, {'task': [30, 30], 'done_mdp': False},
                 {'task': [31, 31], 'done_mdp': False}, {'task': [32, 32], 'done_mdp': False},{'task': [33, 33], 'done_mdp': False}, {'task': [34, 34], 'done_mdp': False},{'task': [35, 35], 'done_mdp': False}, {'task': [36, 36], 'done_mdp': False},{'task': [37, 37], 'done_mdp': False}, {'task': [38, 38], 'done_mdp': False},{'task': [39, 39], 'done_mdp': False}, {'task': [40, 40], 'done_mdp': False},
                 {'task': [41, 41], 'done_mdp': False}, {'task': [42, 42], 'done_mdp': False},{'task': [43, 43], 'done_mdp': False}, {'task': [44, 44], 'done_mdp': False},{'task': [45, 45], 'done_mdp': False}, {'task': [46, 46], 'done_mdp': False},{'task': [47, 47], 'done_mdp': False}, {'task': [48, 48], 'done_mdp': False})

# --------------------------- 2570 ---------------------------
'''
No 0
roadnet_3_3.json
anon_3_3_300_0.3_bi1_2570.json
'''
dic_traffic_env_conf_3_3_2570 = {
        "ADJACENCY_BY_CONNECTION_OR_GEO": True,
        "USE_LANE_ADJACENCY": True,
        "TRAFFIC_FILE": "anon_3_3_300_0.3_bi1_2570.json",
        "THREADNUM": 8,
        "SAVEREPLAY": False,
        "RLTRAFFICLIGHT": True,
        "INTERVAL": 1,
        "NUM_INTERSECTIONS": 9,  # attention
        "ACTION_PATTERN": "switch",  # set
        "MEASURE_TIME": 10,
        "MIN_ACTION_TIME": 5,  # 10
        "YELLOW_TIME": 5,
        "ALL_RED_TIME": 0,
        "DEBUG": False,  # False
        "BINARY_PHASE_EXPANSION": True,
        "FAST_COMPUTE": True,
        'NUM_AGENTS': 1,

        "NEIGHBOR": False,
        "MODEL_NAME": "STGAT",
        "SIMULATOR_TYPE": "anon",
        "TOP_K_ADJACENCY": 9,
        "TOP_K_ADJACENCY_LANE": 6,

        "SAVEREPLAY": False,
        "NUM_ROW": 3,  # attention
        "NUM_COL": 3,  # attention

        "VOLUME": 300,
        "ROADNET_FILE": "roadnet_{0}.json".format("3_3"),  # attention

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
            # "adjacency_matrix_lane",
        ],

        "DIC_FEATURE_DIM": dict(
            D_LANE_QUEUE_LENGTH=(4,),
            D_LANE_NUM_VEHICLE=(4,),

            D_COMING_VEHICLE=(12,),
            D_LEAVING_VEHICLE=(12,),

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
            D_ADJACENCY_MATRIX_LANE=(6,)

        ),

        "DIC_REWARD_INFO": {
            "flickering": 0,
            "sum_lane_queue_length": 0,
            "sum_lane_wait_time": 0,
            "sum_lane_num_vehicle_left": 0,
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
                0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
                1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
                2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
                3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
            },
            "anon": {
                # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
                2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
                3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
                # 'WSWL',
                # 'ESEL',
                # 'WSES',
                # 'NSSS',
                # 'NSNL',
                # 'SSSL',
            },
        }
    }
infos_3_3 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False},
                 {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False},
                 {'task': [5, 5], 'done_mdp': False}, {'task': [6, 6], 'done_mdp': False},
                 {'task': [7, 7], 'done_mdp': False}, {'task': [8, 8], 'done_mdp': False},
                 {'task': [9, 9], 'done_mdp': False})


'''
No 1
roadnet_6_6.json
anon_6_6_300_0.3_bi_2570.json
'''
dic_traffic_env_conf_6_6_2570 = {
        "ADJACENCY_BY_CONNECTION_OR_GEO": True,
        "USE_LANE_ADJACENCY": True,
        "TRAFFIC_FILE": "anon_6_6_300_0.3_bi_2570.json",
        "THREADNUM": 8,
        "SAVEREPLAY": False,
        "RLTRAFFICLIGHT": True,
        "INTERVAL": 1,
        "NUM_INTERSECTIONS": 36,  # attention
        "ACTION_PATTERN": "switch",  # set
        "MEASURE_TIME": 10,
        "MIN_ACTION_TIME": 5,  # 10
        "YELLOW_TIME": 5,
        "ALL_RED_TIME": 0,
        "DEBUG": False,  # False
        "BINARY_PHASE_EXPANSION": True,
        "FAST_COMPUTE": True,
        'NUM_AGENTS': 1,

        "NEIGHBOR": False,
        "MODEL_NAME": "STGAT",
        "SIMULATOR_TYPE": "anon",
        "TOP_K_ADJACENCY": 9,
        "TOP_K_ADJACENCY_LANE": 6,

        "SAVEREPLAY": False,
        "NUM_ROW": 6,  # attention
        "NUM_COL": 6,  # attention

        "VOLUME": 300,
        "ROADNET_FILE": "roadnet_{0}.json".format("6_6"),  # attention

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
            # "adjacency_matrix_lane",
        ],

        "DIC_FEATURE_DIM": dict(
            D_LANE_QUEUE_LENGTH=(4,),
            D_LANE_NUM_VEHICLE=(4,),

            D_COMING_VEHICLE=(12,),
            D_LEAVING_VEHICLE=(12,),

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
            D_ADJACENCY_MATRIX_LANE=(6,)

        ),

        "DIC_REWARD_INFO": {
            "flickering": 0,
            "sum_lane_queue_length": 0,
            "sum_lane_wait_time": 0,
            "sum_lane_num_vehicle_left": 0,
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
                0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
                1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
                2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
                3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
            },
            "anon": {
                # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
                2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
                3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
                # 'WSWL',
                # 'ESEL',
                # 'WSES',
                # 'NSSS',
                # 'NSNL',
                # 'SSSL',
            },
        }
    }
infos_6_6 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False},
                 {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False},
                 {'task': [5, 5], 'done_mdp': False}, {'task': [6, 6], 'done_mdp': False},
                 {'task': [7, 7], 'done_mdp': False}, {'task': [8, 8], 'done_mdp': False},
                 {'task': [9, 9], 'done_mdp': False}, {'task': [10, 10], 'done_mdp': False},
                 {'task': [11, 11], 'done_mdp': False}, {'task': [12, 12], 'done_mdp': False},
                 {'task': [13, 13], 'done_mdp': False}, {'task': [14, 14], 'done_mdp': False},
                 {'task': [15, 15], 'done_mdp': False}, {'task': [16, 16], 'done_mdp': False},
                 {'task': [17, 17], 'done_mdp': False}, {'task': [18, 18], 'done_mdp': False},
                 {'task': [19, 19], 'done_mdp': False}, {'task': [20, 20], 'done_mdp': False},
                 {'task': [21, 21], 'done_mdp': False}, {'task': [22, 22], 'done_mdp': False},
                 {'task': [23, 23], 'done_mdp': False}, {'task': [24, 24], 'done_mdp': False},
                 {'task': [25, 25], 'done_mdp': False}, {'task': [26, 26], 'done_mdp': False},
                 {'task': [27, 27], 'done_mdp': False}, {'task': [28, 28], 'done_mdp': False},
                 {'task': [29, 29], 'done_mdp': False}, {'task': [30, 30], 'done_mdp': False},
                 {'task': [31, 31], 'done_mdp': False}, {'task': [32, 32], 'done_mdp': False},
                 {'task': [33, 33], 'done_mdp': False}, {'task': [34, 34], 'done_mdp': False},
                 {'task': [35, 35], 'done_mdp': False}, {'task': [36, 36], 'done_mdp': False})


'''
No 2
roadnet_10_10.json
anon_10_10_300_0.3_bi_2570.json
'''
dic_traffic_env_conf_10_10_2570 = {
        "ADJACENCY_BY_CONNECTION_OR_GEO": True,
        "USE_LANE_ADJACENCY": True,
        "TRAFFIC_FILE": "anon_10_10_300_0.3_bi_2570.json",
        "THREADNUM": 8,
        "SAVEREPLAY": False,
        "RLTRAFFICLIGHT": True,
        "INTERVAL": 1,
        "NUM_INTERSECTIONS": 100,  # attention
        "ACTION_PATTERN": "switch",  # set
        "MEASURE_TIME": 10,
        "MIN_ACTION_TIME": 5,  # 10
        "YELLOW_TIME": 5,
        "ALL_RED_TIME": 0,
        "DEBUG": False,  # False
        "BINARY_PHASE_EXPANSION": True,
        "FAST_COMPUTE": True,
        'NUM_AGENTS': 1,

        "NEIGHBOR": False,
        "MODEL_NAME": "STGAT",
        "SIMULATOR_TYPE": "anon",
        "TOP_K_ADJACENCY": 9,
        "TOP_K_ADJACENCY_LANE": 6,

        "SAVEREPLAY": False,
        "NUM_ROW": 10,  # attention
        "NUM_COL": 10,  # attention

        "VOLUME": 300,
        "ROADNET_FILE": "roadnet_{0}.json".format("10_10"),  # attention

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
            # "adjacency_matrix_lane",
        ],

        "DIC_FEATURE_DIM": dict(
            D_LANE_QUEUE_LENGTH=(4,),
            D_LANE_NUM_VEHICLE=(4,),

            D_COMING_VEHICLE=(12,),
            D_LEAVING_VEHICLE=(12,),

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
            D_ADJACENCY_MATRIX_LANE=(6,)

        ),

        "DIC_REWARD_INFO": {
            "flickering": 0,
            "sum_lane_queue_length": 0,
            "sum_lane_wait_time": 0,
            "sum_lane_num_vehicle_left": 0,
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
                0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
                1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
                2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
                3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
            },
            "anon": {
                # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
                2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
                3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
                # 'WSWL',
                # 'ESEL',
                # 'WSES',
                # 'NSSS',
                # 'NSNL',
                # 'SSSL',
            },
        }
    }
infos_10_10 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False}, {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False}, {'task': [5, 5], 'done_mdp': False},{'task': [6, 6], 'done_mdp': False},{'task': [7, 7], 'done_mdp': False},{'task': [8, 8], 'done_mdp': False},{'task': [9, 9], 'done_mdp': False},{'task': [10, 10], 'done_mdp': False},
                 {'task': [11, 11], 'done_mdp': False}, {'task': [12, 12], 'done_mdp': False},{'task': [13, 13], 'done_mdp': False}, {'task': [14, 14], 'done_mdp': False},{'task': [15, 15], 'done_mdp': False}, {'task': [16, 16], 'done_mdp': False},{'task': [17, 17], 'done_mdp': False}, {'task': [18, 18], 'done_mdp': False},{'task': [19, 19], 'done_mdp': False}, {'task': [20, 20], 'done_mdp': False},
                 {'task': [21, 21], 'done_mdp': False}, {'task': [22, 22], 'done_mdp': False},{'task': [23, 23], 'done_mdp': False}, {'task': [24, 24], 'done_mdp': False},{'task': [25, 25], 'done_mdp': False}, {'task': [26, 26], 'done_mdp': False},{'task': [27, 27], 'done_mdp': False}, {'task': [28, 28], 'done_mdp': False},{'task': [29, 29], 'done_mdp': False}, {'task': [30, 30], 'done_mdp': False},
                 {'task': [31, 31], 'done_mdp': False}, {'task': [32, 32], 'done_mdp': False},{'task': [33, 33], 'done_mdp': False}, {'task': [34, 34], 'done_mdp': False},{'task': [35, 35], 'done_mdp': False}, {'task': [36, 36], 'done_mdp': False},{'task': [37, 37], 'done_mdp': False}, {'task': [38, 38], 'done_mdp': False},{'task': [39, 39], 'done_mdp': False}, {'task': [40, 40], 'done_mdp': False},
                 {'task': [41, 41], 'done_mdp': False}, {'task': [42, 42], 'done_mdp': False},{'task': [43, 43], 'done_mdp': False}, {'task': [44, 44], 'done_mdp': False},{'task': [45, 45], 'done_mdp': False}, {'task': [46, 46], 'done_mdp': False},{'task': [47, 47], 'done_mdp': False}, {'task': [48, 48], 'done_mdp': False},{'task': [49, 49], 'done_mdp': False}, {'task': [50, 50], 'done_mdp': False},
                 {'task': [51, 51], 'done_mdp': False}, {'task': [52, 52], 'done_mdp': False},{'task': [53, 53], 'done_mdp': False}, {'task': [54, 54], 'done_mdp': False},{'task': [55, 55], 'done_mdp': False}, {'task': [56, 56], 'done_mdp': False},{'task': [57, 57], 'done_mdp': False}, {'task': [58, 58], 'done_mdp': False},{'task': [59, 59], 'done_mdp': False}, {'task': [60, 60], 'done_mdp': False},
                 {'task': [61, 61], 'done_mdp': False}, {'task': [62, 62], 'done_mdp': False},{'task': [63, 63], 'done_mdp': False}, {'task': [64, 64], 'done_mdp': False},{'task': [65, 65], 'done_mdp': False}, {'task': [66, 66], 'done_mdp': False},{'task': [67, 67], 'done_mdp': False}, {'task': [68, 68], 'done_mdp': False},{'task': [69, 69], 'done_mdp': False}, {'task': [70, 70], 'done_mdp': False},
                 {'task': [71, 71], 'done_mdp': False}, {'task': [72, 72], 'done_mdp': False},{'task': [73, 73], 'done_mdp': False}, {'task': [74, 74], 'done_mdp': False},{'task': [75, 75], 'done_mdp': False}, {'task': [76, 76], 'done_mdp': False},{'task': [77, 77], 'done_mdp': False}, {'task': [78, 78], 'done_mdp': False},{'task': [79, 79], 'done_mdp': False}, {'task': [80, 80], 'done_mdp': False},
                 {'task': [81, 81], 'done_mdp': False}, {'task': [82, 82], 'done_mdp': False},{'task': [83, 83], 'done_mdp': False}, {'task': [84, 84], 'done_mdp': False},{'task': [85, 85], 'done_mdp': False}, {'task': [86, 86], 'done_mdp': False},{'task': [87, 87], 'done_mdp': False}, {'task': [88, 88], 'done_mdp': False},{'task': [89, 89], 'done_mdp': False}, {'task': [90, 90], 'done_mdp': False},
                 {'task': [91, 91], 'done_mdp': False}, {'task': [92, 92], 'done_mdp': False},{'task': [93, 93], 'done_mdp': False}, {'task': [94, 94], 'done_mdp': False},{'task': [95, 95], 'done_mdp': False}, {'task': [96, 96], 'done_mdp': False},{'task': [97, 97], 'done_mdp': False}, {'task': [98, 98], 'done_mdp': False},{'task': [99, 99], 'done_mdp': False}, {'task': [100, 100], 'done_mdp': False},)


'''
No 3
roadnet_3_4.json
anon_3_4_jinan_real_2570.json
'''
dic_traffic_env_conf_3_4_2570 = {
        "ADJACENCY_BY_CONNECTION_OR_GEO": True,
        "USE_LANE_ADJACENCY": True,
        "TRAFFIC_FILE": "anon_3_4_jinan_real_2570.json",
        "THREADNUM": 8,
        "SAVEREPLAY": False,
        "RLTRAFFICLIGHT": True,
        "INTERVAL": 1,
        "NUM_INTERSECTIONS": 12,  # attention
        "ACTION_PATTERN": "switch",  # set
        "MEASURE_TIME": 10,
        "MIN_ACTION_TIME": 5,  # 10
        "YELLOW_TIME": 5,
        "ALL_RED_TIME": 0,
        "DEBUG": False,  # False
        "BINARY_PHASE_EXPANSION": True,
        "FAST_COMPUTE": True,
        'NUM_AGENTS': 1,

        "NEIGHBOR": False,
        "MODEL_NAME": "STGAT",
        "SIMULATOR_TYPE": "anon",
        "TOP_K_ADJACENCY": 9,
        "TOP_K_ADJACENCY_LANE": 6,

        "SAVEREPLAY": False,
        "NUM_ROW": 4,  # attention
        "NUM_COL": 3,  # attention

        "VOLUME": 300,
        "ROADNET_FILE": "roadnet_{0}.json".format("3_4"),  # attention

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
            # "adjacency_matrix_lane",
        ],

        "DIC_FEATURE_DIM": dict(
            D_LANE_QUEUE_LENGTH=(4,),
            D_LANE_NUM_VEHICLE=(4,),

            D_COMING_VEHICLE=(12,),
            D_LEAVING_VEHICLE=(12,),

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
            D_ADJACENCY_MATRIX_LANE=(6,)

        ),

        "DIC_REWARD_INFO": {
            "flickering": 0,
            "sum_lane_queue_length": 0,
            "sum_lane_wait_time": 0,
            "sum_lane_num_vehicle_left": 0,
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
                0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
                1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
                2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
                3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
            },
            "anon": {
                # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
                2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
                3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
                # 'WSWL',
                # 'ESEL',
                # 'WSES',
                # 'NSSS',
                # 'NSNL',
                # 'SSSL',
            },
        }
    }
infos_3_4 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False}, {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False}, {'task': [5, 5], 'done_mdp': False},{'task': [6, 6], 'done_mdp': False},{'task': [7, 7], 'done_mdp': False},{'task': [8, 8], 'done_mdp': False},{'task': [9, 9], 'done_mdp': False},{'task': [10, 10], 'done_mdp': False},
                 {'task': [11, 11], 'done_mdp': False}, {'task': [12, 12], 'done_mdp': False})

'''
No 4
roadnet_4_4.json
anon_4_4_hangzhou_real_2570.json
'''
dic_traffic_env_conf_4_4_2570 = {
    "ADJACENCY_BY_CONNECTION_OR_GEO": True,
    "USE_LANE_ADJACENCY": True,
    "TRAFFIC_FILE": "anon_4_4_hangzhou_real_2570.json",  # anon_4_4_hangzhou_real.json anon_3_3_300_0.3_bi.json# attention
    "THREADNUM": 8,
    "SAVEREPLAY": False,
    "RLTRAFFICLIGHT": True,
    "INTERVAL": 1,
    "NUM_INTERSECTIONS": 16,  # attention
    "ACTION_PATTERN": "switch",  # set
    "MEASURE_TIME": 10,
    "MIN_ACTION_TIME": 5,  # 10
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "DEBUG": False,  # False
    "BINARY_PHASE_EXPANSION": True,
    "FAST_COMPUTE": True,
    'NUM_AGENTS': 1,

    "NEIGHBOR": False,
    "MODEL_NAME": "STGAT",
    "SIMULATOR_TYPE": "anon",
    "TOP_K_ADJACENCY": 9,
    "TOP_K_ADJACENCY_LANE": 6,

    "SAVEREPLAY": False,
    "NUM_ROW": 4,  # attention
    "NUM_COL": 4,  # attention

    "VOLUME": 300,
    "ROADNET_FILE": "roadnet_{0}.json".format("4_4"),  # attention

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
        # "adjacency_matrix_lane",
    ],

    "DIC_FEATURE_DIM": dict(
        D_LANE_QUEUE_LENGTH=(4,),
        D_LANE_NUM_VEHICLE=(4,),

        D_COMING_VEHICLE=(12,),
        D_LEAVING_VEHICLE=(12,),

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
        D_ADJACENCY_MATRIX_LANE=(6,)

    ),

    "DIC_REWARD_INFO": {
        "flickering": 0,
        "sum_lane_queue_length": 0,
        "sum_lane_wait_time": 0,
        "sum_lane_num_vehicle_left": 0,
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
            0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
            1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
            2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
            3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
        },
        "anon": {
            # 0: [0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
            2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
            3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
            4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
            # 'WSWL',
            # 'ESEL',
            # 'WSES',
            # 'NSSS',
            # 'NSNL',
            # 'SSSL',
        },
    }
}
infos_4_4 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False}, {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False}, {'task': [5, 5], 'done_mdp': False},{'task': [6, 6], 'done_mdp': False},{'task': [7, 7], 'done_mdp': False},{'task': [8, 8], 'done_mdp': False},{'task': [9, 9], 'done_mdp': False},{'task': [10, 10], 'done_mdp': False},
                 {'task': [11, 11], 'done_mdp': False}, {'task': [12, 12], 'done_mdp': False},{'task': [13, 13], 'done_mdp': False}, {'task': [14, 14], 'done_mdp': False},{'task': [15, 15], 'done_mdp': False}, {'task': [16, 16], 'done_mdp': False})


'''
No 5
roadnet_16_3.json
anon_16_3_newyork_real_2570.json
'''
dic_traffic_env_conf_16_3_2570 = {
    "ADJACENCY_BY_CONNECTION_OR_GEO": True,
    "USE_LANE_ADJACENCY": True,
    "TRAFFIC_FILE": "anon_16_3_newyork_real_2570.json",  # anon_4_4_hangzhou_real.json anon_3_3_300_0.3_bi.json# attention
    "THREADNUM": 8,
    "SAVEREPLAY": False,
    "RLTRAFFICLIGHT": True,
    "INTERVAL": 1,
    "NUM_INTERSECTIONS": 48,  # attention
    "ACTION_PATTERN": "switch",  # set
    "MEASURE_TIME": 10,
    "MIN_ACTION_TIME": 5,  # 10
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "DEBUG": False,  # False
    "BINARY_PHASE_EXPANSION": True,
    "FAST_COMPUTE": True,
    'NUM_AGENTS': 1,

    "NEIGHBOR": False,
    "MODEL_NAME": "STGAT",
    "SIMULATOR_TYPE": "anon",
    "TOP_K_ADJACENCY": 9,
    "TOP_K_ADJACENCY_LANE": 6,

    "SAVEREPLAY": False,
    "NUM_ROW": 3,  # attention
    "NUM_COL": 16,  # attention

    "VOLUME": 300,
    "ROADNET_FILE": "roadnet_{0}.json".format("16_3"),  # attention

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
        # "adjacency_matrix_lane",
    ],

    "DIC_FEATURE_DIM": dict(
        D_LANE_QUEUE_LENGTH=(4,),
        D_LANE_NUM_VEHICLE=(4,),

        D_COMING_VEHICLE=(12,),
        D_LEAVING_VEHICLE=(12,),

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
        D_ADJACENCY_MATRIX_LANE=(6,)

    ),

    "DIC_REWARD_INFO": {
        "flickering": 0,
        "sum_lane_queue_length": 0,
        "sum_lane_wait_time": 0,
        "sum_lane_num_vehicle_left": 0,
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
            0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
            1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
            2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
            3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
        },
        "anon": {
            # 0: [0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
            2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
            3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
            4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
            # 'WSWL',
            # 'ESEL',
            # 'WSES',
            # 'NSSS',
            # 'NSNL',
            # 'SSSL',
        },
    }
}
infos_16_3 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False}, {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False}, {'task': [5, 5], 'done_mdp': False},{'task': [6, 6], 'done_mdp': False},{'task': [7, 7], 'done_mdp': False},{'task': [8, 8], 'done_mdp': False},{'task': [9, 9], 'done_mdp': False},{'task': [10, 10], 'done_mdp': False},
                 {'task': [11, 11], 'done_mdp': False}, {'task': [12, 12], 'done_mdp': False},{'task': [13, 13], 'done_mdp': False}, {'task': [14, 14], 'done_mdp': False},{'task': [15, 15], 'done_mdp': False}, {'task': [16, 16], 'done_mdp': False},{'task': [17, 17], 'done_mdp': False}, {'task': [18, 18], 'done_mdp': False},{'task': [19, 19], 'done_mdp': False}, {'task': [20, 20], 'done_mdp': False},
                 {'task': [21, 21], 'done_mdp': False}, {'task': [22, 22], 'done_mdp': False},{'task': [23, 23], 'done_mdp': False}, {'task': [24, 24], 'done_mdp': False},{'task': [25, 25], 'done_mdp': False}, {'task': [26, 26], 'done_mdp': False},{'task': [27, 27], 'done_mdp': False}, {'task': [28, 28], 'done_mdp': False},{'task': [29, 29], 'done_mdp': False}, {'task': [30, 30], 'done_mdp': False},
                 {'task': [31, 31], 'done_mdp': False}, {'task': [32, 32], 'done_mdp': False},{'task': [33, 33], 'done_mdp': False}, {'task': [34, 34], 'done_mdp': False},{'task': [35, 35], 'done_mdp': False}, {'task': [36, 36], 'done_mdp': False},{'task': [37, 37], 'done_mdp': False}, {'task': [38, 38], 'done_mdp': False},{'task': [39, 39], 'done_mdp': False}, {'task': [40, 40], 'done_mdp': False},
                 {'task': [41, 41], 'done_mdp': False}, {'task': [42, 42], 'done_mdp': False},{'task': [43, 43], 'done_mdp': False}, {'task': [44, 44], 'done_mdp': False},{'task': [45, 45], 'done_mdp': False}, {'task': [46, 46], 'done_mdp': False},{'task': [47, 47], 'done_mdp': False}, {'task': [48, 48], 'done_mdp': False})


# --------------------------- 4770 ---------------------------
'''
No 0
roadnet_3_3.json
anon_3_3_300_0.3_bi1_4770.json
'''
dic_traffic_env_conf_3_3_4770 = {
        "ADJACENCY_BY_CONNECTION_OR_GEO": True,
        "USE_LANE_ADJACENCY": True,
        "TRAFFIC_FILE": "anon_3_3_300_0.3_bi1_4770.json",
        "THREADNUM": 8,
        "SAVEREPLAY": False,
        "RLTRAFFICLIGHT": True,
        "INTERVAL": 1,
        "NUM_INTERSECTIONS": 9,  # attention
        "ACTION_PATTERN": "switch",  # set
        "MEASURE_TIME": 10,
        "MIN_ACTION_TIME": 5,  # 10
        "YELLOW_TIME": 5,
        "ALL_RED_TIME": 0,
        "DEBUG": False,  # False
        "BINARY_PHASE_EXPANSION": True,
        "FAST_COMPUTE": True,
        'NUM_AGENTS': 1,

        "NEIGHBOR": False,
        "MODEL_NAME": "STGAT",
        "SIMULATOR_TYPE": "anon",
        "TOP_K_ADJACENCY": 9,
        "TOP_K_ADJACENCY_LANE": 6,

        "SAVEREPLAY": False,
        "NUM_ROW": 3,  # attention
        "NUM_COL": 3,  # attention

        "VOLUME": 300,
        "ROADNET_FILE": "roadnet_{0}.json".format("3_3"),  # attention

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
            # "adjacency_matrix_lane",
        ],

        "DIC_FEATURE_DIM": dict(
            D_LANE_QUEUE_LENGTH=(4,),
            D_LANE_NUM_VEHICLE=(4,),

            D_COMING_VEHICLE=(12,),
            D_LEAVING_VEHICLE=(12,),

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
            D_ADJACENCY_MATRIX_LANE=(6,)

        ),

        "DIC_REWARD_INFO": {
            "flickering": 0,
            "sum_lane_queue_length": 0,
            "sum_lane_wait_time": 0,
            "sum_lane_num_vehicle_left": 0,
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
                0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
                1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
                2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
                3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
            },
            "anon": {
                # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
                2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
                3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
                # 'WSWL',
                # 'ESEL',
                # 'WSES',
                # 'NSSS',
                # 'NSNL',
                # 'SSSL',
            },
        }
    }
infos_3_3 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False},
                 {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False},
                 {'task': [5, 5], 'done_mdp': False}, {'task': [6, 6], 'done_mdp': False},
                 {'task': [7, 7], 'done_mdp': False}, {'task': [8, 8], 'done_mdp': False},
                 {'task': [9, 9], 'done_mdp': False})


'''
No 1
roadnet_6_6.json
anon_6_6_300_0.3_bi_4770.json
'''
dic_traffic_env_conf_6_6_4770 = {
        "ADJACENCY_BY_CONNECTION_OR_GEO": True,
        "USE_LANE_ADJACENCY": True,
        "TRAFFIC_FILE": "anon_6_6_300_0.3_bi_4770.json",
        "THREADNUM": 8,
        "SAVEREPLAY": False,
        "RLTRAFFICLIGHT": True,
        "INTERVAL": 1,
        "NUM_INTERSECTIONS": 36,  # attention
        "ACTION_PATTERN": "switch",  # set
        "MEASURE_TIME": 10,
        "MIN_ACTION_TIME": 5,  # 10
        "YELLOW_TIME": 5,
        "ALL_RED_TIME": 0,
        "DEBUG": False,  # False
        "BINARY_PHASE_EXPANSION": True,
        "FAST_COMPUTE": True,
        'NUM_AGENTS': 1,

        "NEIGHBOR": False,
        "MODEL_NAME": "STGAT",
        "SIMULATOR_TYPE": "anon",
        "TOP_K_ADJACENCY": 9,
        "TOP_K_ADJACENCY_LANE": 6,

        "SAVEREPLAY": False,
        "NUM_ROW": 6,  # attention
        "NUM_COL": 6,  # attention

        "VOLUME": 300,
        "ROADNET_FILE": "roadnet_{0}.json".format("6_6"),  # attention

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
            # "adjacency_matrix_lane",
        ],

        "DIC_FEATURE_DIM": dict(
            D_LANE_QUEUE_LENGTH=(4,),
            D_LANE_NUM_VEHICLE=(4,),

            D_COMING_VEHICLE=(12,),
            D_LEAVING_VEHICLE=(12,),

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
            D_ADJACENCY_MATRIX_LANE=(6,)

        ),

        "DIC_REWARD_INFO": {
            "flickering": 0,
            "sum_lane_queue_length": 0,
            "sum_lane_wait_time": 0,
            "sum_lane_num_vehicle_left": 0,
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
                0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
                1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
                2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
                3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
            },
            "anon": {
                # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
                2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
                3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
                # 'WSWL',
                # 'ESEL',
                # 'WSES',
                # 'NSSS',
                # 'NSNL',
                # 'SSSL',
            },
        }
    }
infos_6_6 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False},
                 {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False},
                 {'task': [5, 5], 'done_mdp': False}, {'task': [6, 6], 'done_mdp': False},
                 {'task': [7, 7], 'done_mdp': False}, {'task': [8, 8], 'done_mdp': False},
                 {'task': [9, 9], 'done_mdp': False}, {'task': [10, 10], 'done_mdp': False},
                 {'task': [11, 11], 'done_mdp': False}, {'task': [12, 12], 'done_mdp': False},
                 {'task': [13, 13], 'done_mdp': False}, {'task': [14, 14], 'done_mdp': False},
                 {'task': [15, 15], 'done_mdp': False}, {'task': [16, 16], 'done_mdp': False},
                 {'task': [17, 17], 'done_mdp': False}, {'task': [18, 18], 'done_mdp': False},
                 {'task': [19, 19], 'done_mdp': False}, {'task': [20, 20], 'done_mdp': False},
                 {'task': [21, 21], 'done_mdp': False}, {'task': [22, 22], 'done_mdp': False},
                 {'task': [23, 23], 'done_mdp': False}, {'task': [24, 24], 'done_mdp': False},
                 {'task': [25, 25], 'done_mdp': False}, {'task': [26, 26], 'done_mdp': False},
                 {'task': [27, 27], 'done_mdp': False}, {'task': [28, 28], 'done_mdp': False},
                 {'task': [29, 29], 'done_mdp': False}, {'task': [30, 30], 'done_mdp': False},
                 {'task': [31, 31], 'done_mdp': False}, {'task': [32, 32], 'done_mdp': False},
                 {'task': [33, 33], 'done_mdp': False}, {'task': [34, 34], 'done_mdp': False},
                 {'task': [35, 35], 'done_mdp': False}, {'task': [36, 36], 'done_mdp': False})


'''
No 2
roadnet_10_10.json
anon_10_10_300_0.3_bi_4770.json
'''
dic_traffic_env_conf_10_10_4770 = {
        "ADJACENCY_BY_CONNECTION_OR_GEO": True,
        "USE_LANE_ADJACENCY": True,
        "TRAFFIC_FILE": "anon_10_10_300_0.3_bi_4770.json",
        "THREADNUM": 8,
        "SAVEREPLAY": False,
        "RLTRAFFICLIGHT": True,
        "INTERVAL": 1,
        "NUM_INTERSECTIONS": 100,  # attention
        "ACTION_PATTERN": "switch",  # set
        "MEASURE_TIME": 10,
        "MIN_ACTION_TIME": 5,  # 10
        "YELLOW_TIME": 5,
        "ALL_RED_TIME": 0,
        "DEBUG": False,  # False
        "BINARY_PHASE_EXPANSION": True,
        "FAST_COMPUTE": True,
        'NUM_AGENTS': 1,

        "NEIGHBOR": False,
        "MODEL_NAME": "STGAT",
        "SIMULATOR_TYPE": "anon",
        "TOP_K_ADJACENCY": 9,
        "TOP_K_ADJACENCY_LANE": 6,

        "SAVEREPLAY": False,
        "NUM_ROW": 10,  # attention
        "NUM_COL": 10,  # attention

        "VOLUME": 300,
        "ROADNET_FILE": "roadnet_{0}.json".format("10_10"),  # attention

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
            # "adjacency_matrix_lane",
        ],

        "DIC_FEATURE_DIM": dict(
            D_LANE_QUEUE_LENGTH=(4,),
            D_LANE_NUM_VEHICLE=(4,),

            D_COMING_VEHICLE=(12,),
            D_LEAVING_VEHICLE=(12,),

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
            D_ADJACENCY_MATRIX_LANE=(6,)

        ),

        "DIC_REWARD_INFO": {
            "flickering": 0,
            "sum_lane_queue_length": 0,
            "sum_lane_wait_time": 0,
            "sum_lane_num_vehicle_left": 0,
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
                0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
                1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
                2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
                3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
            },
            "anon": {
                # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
                2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
                3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
                # 'WSWL',
                # 'ESEL',
                # 'WSES',
                # 'NSSS',
                # 'NSNL',
                # 'SSSL',
            },
        }
    }
infos_10_10 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False}, {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False}, {'task': [5, 5], 'done_mdp': False},{'task': [6, 6], 'done_mdp': False},{'task': [7, 7], 'done_mdp': False},{'task': [8, 8], 'done_mdp': False},{'task': [9, 9], 'done_mdp': False},{'task': [10, 10], 'done_mdp': False},
                 {'task': [11, 11], 'done_mdp': False}, {'task': [12, 12], 'done_mdp': False},{'task': [13, 13], 'done_mdp': False}, {'task': [14, 14], 'done_mdp': False},{'task': [15, 15], 'done_mdp': False}, {'task': [16, 16], 'done_mdp': False},{'task': [17, 17], 'done_mdp': False}, {'task': [18, 18], 'done_mdp': False},{'task': [19, 19], 'done_mdp': False}, {'task': [20, 20], 'done_mdp': False},
                 {'task': [21, 21], 'done_mdp': False}, {'task': [22, 22], 'done_mdp': False},{'task': [23, 23], 'done_mdp': False}, {'task': [24, 24], 'done_mdp': False},{'task': [25, 25], 'done_mdp': False}, {'task': [26, 26], 'done_mdp': False},{'task': [27, 27], 'done_mdp': False}, {'task': [28, 28], 'done_mdp': False},{'task': [29, 29], 'done_mdp': False}, {'task': [30, 30], 'done_mdp': False},
                 {'task': [31, 31], 'done_mdp': False}, {'task': [32, 32], 'done_mdp': False},{'task': [33, 33], 'done_mdp': False}, {'task': [34, 34], 'done_mdp': False},{'task': [35, 35], 'done_mdp': False}, {'task': [36, 36], 'done_mdp': False},{'task': [37, 37], 'done_mdp': False}, {'task': [38, 38], 'done_mdp': False},{'task': [39, 39], 'done_mdp': False}, {'task': [40, 40], 'done_mdp': False},
                 {'task': [41, 41], 'done_mdp': False}, {'task': [42, 42], 'done_mdp': False},{'task': [43, 43], 'done_mdp': False}, {'task': [44, 44], 'done_mdp': False},{'task': [45, 45], 'done_mdp': False}, {'task': [46, 46], 'done_mdp': False},{'task': [47, 47], 'done_mdp': False}, {'task': [48, 48], 'done_mdp': False},{'task': [49, 49], 'done_mdp': False}, {'task': [50, 50], 'done_mdp': False},
                 {'task': [51, 51], 'done_mdp': False}, {'task': [52, 52], 'done_mdp': False},{'task': [53, 53], 'done_mdp': False}, {'task': [54, 54], 'done_mdp': False},{'task': [55, 55], 'done_mdp': False}, {'task': [56, 56], 'done_mdp': False},{'task': [57, 57], 'done_mdp': False}, {'task': [58, 58], 'done_mdp': False},{'task': [59, 59], 'done_mdp': False}, {'task': [60, 60], 'done_mdp': False},
                 {'task': [61, 61], 'done_mdp': False}, {'task': [62, 62], 'done_mdp': False},{'task': [63, 63], 'done_mdp': False}, {'task': [64, 64], 'done_mdp': False},{'task': [65, 65], 'done_mdp': False}, {'task': [66, 66], 'done_mdp': False},{'task': [67, 67], 'done_mdp': False}, {'task': [68, 68], 'done_mdp': False},{'task': [69, 69], 'done_mdp': False}, {'task': [70, 70], 'done_mdp': False},
                 {'task': [71, 71], 'done_mdp': False}, {'task': [72, 72], 'done_mdp': False},{'task': [73, 73], 'done_mdp': False}, {'task': [74, 74], 'done_mdp': False},{'task': [75, 75], 'done_mdp': False}, {'task': [76, 76], 'done_mdp': False},{'task': [77, 77], 'done_mdp': False}, {'task': [78, 78], 'done_mdp': False},{'task': [79, 79], 'done_mdp': False}, {'task': [80, 80], 'done_mdp': False},
                 {'task': [81, 81], 'done_mdp': False}, {'task': [82, 82], 'done_mdp': False},{'task': [83, 83], 'done_mdp': False}, {'task': [84, 84], 'done_mdp': False},{'task': [85, 85], 'done_mdp': False}, {'task': [86, 86], 'done_mdp': False},{'task': [87, 87], 'done_mdp': False}, {'task': [88, 88], 'done_mdp': False},{'task': [89, 89], 'done_mdp': False}, {'task': [90, 90], 'done_mdp': False},
                 {'task': [91, 91], 'done_mdp': False}, {'task': [92, 92], 'done_mdp': False},{'task': [93, 93], 'done_mdp': False}, {'task': [94, 94], 'done_mdp': False},{'task': [95, 95], 'done_mdp': False}, {'task': [96, 96], 'done_mdp': False},{'task': [97, 97], 'done_mdp': False}, {'task': [98, 98], 'done_mdp': False},{'task': [99, 99], 'done_mdp': False}, {'task': [100, 100], 'done_mdp': False},)


'''
No 3
roadnet_3_4.json
anon_3_4_jinan_real_4770.json
'''
dic_traffic_env_conf_3_4_4770 = {
        "ADJACENCY_BY_CONNECTION_OR_GEO": True,
        "USE_LANE_ADJACENCY": True,
        "TRAFFIC_FILE": "anon_3_4_jinan_real_4770.json",
        "THREADNUM": 8,
        "SAVEREPLAY": False,
        "RLTRAFFICLIGHT": True,
        "INTERVAL": 1,
        "NUM_INTERSECTIONS": 12,  # attention
        "ACTION_PATTERN": "switch",  # set
        "MEASURE_TIME": 10,
        "MIN_ACTION_TIME": 5,  # 10
        "YELLOW_TIME": 5,
        "ALL_RED_TIME": 0,
        "DEBUG": False,  # False
        "BINARY_PHASE_EXPANSION": True,
        "FAST_COMPUTE": True,
        'NUM_AGENTS': 1,

        "NEIGHBOR": False,
        "MODEL_NAME": "STGAT",
        "SIMULATOR_TYPE": "anon",
        "TOP_K_ADJACENCY": 9,
        "TOP_K_ADJACENCY_LANE": 6,

        "SAVEREPLAY": False,
        "NUM_ROW": 4,  # attention
        "NUM_COL": 3,  # attention

        "VOLUME": 300,
        "ROADNET_FILE": "roadnet_{0}.json".format("3_4"),  # attention

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
            # "adjacency_matrix_lane",
        ],

        "DIC_FEATURE_DIM": dict(
            D_LANE_QUEUE_LENGTH=(4,),
            D_LANE_NUM_VEHICLE=(4,),

            D_COMING_VEHICLE=(12,),
            D_LEAVING_VEHICLE=(12,),

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
            D_ADJACENCY_MATRIX_LANE=(6,)

        ),

        "DIC_REWARD_INFO": {
            "flickering": 0,
            "sum_lane_queue_length": 0,
            "sum_lane_wait_time": 0,
            "sum_lane_num_vehicle_left": 0,
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
                0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
                1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
                2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
                3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
            },
            "anon": {
                # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
                2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
                3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
                4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
                # 'WSWL',
                # 'ESEL',
                # 'WSES',
                # 'NSSS',
                # 'NSNL',
                # 'SSSL',
            },
        }
    }
infos_3_4 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False}, {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False}, {'task': [5, 5], 'done_mdp': False},{'task': [6, 6], 'done_mdp': False},{'task': [7, 7], 'done_mdp': False},{'task': [8, 8], 'done_mdp': False},{'task': [9, 9], 'done_mdp': False},{'task': [10, 10], 'done_mdp': False},
                 {'task': [11, 11], 'done_mdp': False}, {'task': [12, 12], 'done_mdp': False})

'''
No 4
roadnet_4_4.json
anon_4_4_hangzhou_real_4770.json
'''
dic_traffic_env_conf_4_4_4770 = {
    "ADJACENCY_BY_CONNECTION_OR_GEO": True,
    "USE_LANE_ADJACENCY": True,
    "TRAFFIC_FILE": "anon_4_4_hangzhou_real_4770.json",  # anon_4_4_hangzhou_real.json anon_3_3_300_0.3_bi.json# attention
    "THREADNUM": 8,
    "SAVEREPLAY": False,
    "RLTRAFFICLIGHT": True,
    "INTERVAL": 1,
    "NUM_INTERSECTIONS": 16,  # attention
    "ACTION_PATTERN": "switch",  # set
    "MEASURE_TIME": 10,
    "MIN_ACTION_TIME": 5,  # 10
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "DEBUG": False,  # False
    "BINARY_PHASE_EXPANSION": True,
    "FAST_COMPUTE": True,
    'NUM_AGENTS': 1,

    "NEIGHBOR": False,
    "MODEL_NAME": "STGAT",
    "SIMULATOR_TYPE": "anon",
    "TOP_K_ADJACENCY": 9,
    "TOP_K_ADJACENCY_LANE": 6,

    "SAVEREPLAY": False,
    "NUM_ROW": 4,  # attention
    "NUM_COL": 4,  # attention

    "VOLUME": 300,
    "ROADNET_FILE": "roadnet_{0}.json".format("4_4"),  # attention

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
        # "adjacency_matrix_lane",
    ],

    "DIC_FEATURE_DIM": dict(
        D_LANE_QUEUE_LENGTH=(4,),
        D_LANE_NUM_VEHICLE=(4,),

        D_COMING_VEHICLE=(12,),
        D_LEAVING_VEHICLE=(12,),

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
        D_ADJACENCY_MATRIX_LANE=(6,)

    ),

    "DIC_REWARD_INFO": {
        "flickering": 0,
        "sum_lane_queue_length": 0,
        "sum_lane_wait_time": 0,
        "sum_lane_num_vehicle_left": 0,
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
            0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
            1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
            2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
            3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
        },
        "anon": {
            # 0: [0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
            2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
            3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
            4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
            # 'WSWL',
            # 'ESEL',
            # 'WSES',
            # 'NSSS',
            # 'NSNL',
            # 'SSSL',
        },
    }
}
infos_4_4 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False}, {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False}, {'task': [5, 5], 'done_mdp': False},{'task': [6, 6], 'done_mdp': False},{'task': [7, 7], 'done_mdp': False},{'task': [8, 8], 'done_mdp': False},{'task': [9, 9], 'done_mdp': False},{'task': [10, 10], 'done_mdp': False},
                 {'task': [11, 11], 'done_mdp': False}, {'task': [12, 12], 'done_mdp': False},{'task': [13, 13], 'done_mdp': False}, {'task': [14, 14], 'done_mdp': False},{'task': [15, 15], 'done_mdp': False}, {'task': [16, 16], 'done_mdp': False})


'''
No 5
roadnet_16_3.json
anon_16_3_newyork_real_4770.json
'''
dic_traffic_env_conf_16_3_4770 = {
    "ADJACENCY_BY_CONNECTION_OR_GEO": True,
    "USE_LANE_ADJACENCY": True,
    "TRAFFIC_FILE": "anon_16_3_newyork_real_4770.json",  # anon_4_4_hangzhou_real.json anon_3_3_300_0.3_bi.json# attention
    "THREADNUM": 8,
    "SAVEREPLAY": False,
    "RLTRAFFICLIGHT": True,
    "INTERVAL": 1,
    "NUM_INTERSECTIONS": 48,  # attention
    "ACTION_PATTERN": "switch",  # set
    "MEASURE_TIME": 10,
    "MIN_ACTION_TIME": 5,  # 10
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "DEBUG": False,  # False
    "BINARY_PHASE_EXPANSION": True,
    "FAST_COMPUTE": True,
    'NUM_AGENTS': 1,

    "NEIGHBOR": False,
    "MODEL_NAME": "STGAT",
    "SIMULATOR_TYPE": "anon",
    "TOP_K_ADJACENCY": 9,
    "TOP_K_ADJACENCY_LANE": 6,

    "SAVEREPLAY": False,
    "NUM_ROW": 3,  # attention
    "NUM_COL": 16,  # attention

    "VOLUME": 300,
    "ROADNET_FILE": "roadnet_{0}.json".format("16_3"),  # attention

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
        # "adjacency_matrix_lane",
    ],

    "DIC_FEATURE_DIM": dict(
        D_LANE_QUEUE_LENGTH=(4,),
        D_LANE_NUM_VEHICLE=(4,),

        D_COMING_VEHICLE=(12,),
        D_LEAVING_VEHICLE=(12,),

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
        D_ADJACENCY_MATRIX_LANE=(6,)

    ),

    "DIC_REWARD_INFO": {
        "flickering": 0,
        "sum_lane_queue_length": 0,
        "sum_lane_wait_time": 0,
        "sum_lane_num_vehicle_left": 0,
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
            0: [0, 1, 0, 1, 0, 0, 0, 0],  # 'WSES',
            1: [0, 0, 0, 0, 0, 1, 0, 1],  # 'NSSS',
            2: [1, 0, 1, 0, 0, 0, 0, 0],  # 'WLEL',
            3: [0, 0, 0, 0, 1, 0, 1, 0]  # 'NLSL',
        },
        "anon": {
            # 0: [0, 0, 0, 0, 0, 0, 0, 0],
            1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
            2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
            3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
            4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
            # 'WSWL',
            # 'ESEL',
            # 'WSES',
            # 'NSSS',
            # 'NSNL',
            # 'SSSL',
        },
    }
}
infos_16_3 = ({'task': [1, 1], 'done_mdp': False}, {'task': [2, 2], 'done_mdp': False}, {'task': [3, 3], 'done_mdp': False}, {'task': [4, 4], 'done_mdp': False}, {'task': [5, 5], 'done_mdp': False},{'task': [6, 6], 'done_mdp': False},{'task': [7, 7], 'done_mdp': False},{'task': [8, 8], 'done_mdp': False},{'task': [9, 9], 'done_mdp': False},{'task': [10, 10], 'done_mdp': False},
                 {'task': [11, 11], 'done_mdp': False}, {'task': [12, 12], 'done_mdp': False},{'task': [13, 13], 'done_mdp': False}, {'task': [14, 14], 'done_mdp': False},{'task': [15, 15], 'done_mdp': False}, {'task': [16, 16], 'done_mdp': False},{'task': [17, 17], 'done_mdp': False}, {'task': [18, 18], 'done_mdp': False},{'task': [19, 19], 'done_mdp': False}, {'task': [20, 20], 'done_mdp': False},
                 {'task': [21, 21], 'done_mdp': False}, {'task': [22, 22], 'done_mdp': False},{'task': [23, 23], 'done_mdp': False}, {'task': [24, 24], 'done_mdp': False},{'task': [25, 25], 'done_mdp': False}, {'task': [26, 26], 'done_mdp': False},{'task': [27, 27], 'done_mdp': False}, {'task': [28, 28], 'done_mdp': False},{'task': [29, 29], 'done_mdp': False}, {'task': [30, 30], 'done_mdp': False},
                 {'task': [31, 31], 'done_mdp': False}, {'task': [32, 32], 'done_mdp': False},{'task': [33, 33], 'done_mdp': False}, {'task': [34, 34], 'done_mdp': False},{'task': [35, 35], 'done_mdp': False}, {'task': [36, 36], 'done_mdp': False},{'task': [37, 37], 'done_mdp': False}, {'task': [38, 38], 'done_mdp': False},{'task': [39, 39], 'done_mdp': False}, {'task': [40, 40], 'done_mdp': False},
                 {'task': [41, 41], 'done_mdp': False}, {'task': [42, 42], 'done_mdp': False},{'task': [43, 43], 'done_mdp': False}, {'task': [44, 44], 'done_mdp': False},{'task': [45, 45], 'done_mdp': False}, {'task': [46, 46], 'done_mdp': False},{'task': [47, 47], 'done_mdp': False}, {'task': [48, 48], 'done_mdp': False})