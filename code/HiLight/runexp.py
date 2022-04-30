# -*- coding: utf-8 -*-

'''

python runexp.py

Run experiments in batch with configuration

'''
import traffic_light_dqn

# ================================= only change these two ========================================
SEED = 3100

setting_memo = "jinan"

# first column: for train, second column: for pre_train
list_traffic_files = [
    [[traffic_light_dqn.TRAFFIC_FILE], [traffic_light_dqn.TRAFFIC_FILE]],
]

list_model_name = [
                   "Deeplight",
                   ]

# ================================= only change these two ========================================
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
from tensorflow import set_random_seed
set_random_seed((SEED))
import json
import os
import time

from sys import platform

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)


PATH_TO_CONF = os.path.join("conf", setting_memo)

for model_name in list_model_name:
    for traffic_file, traffic_file_pretrain in list_traffic_files:
        dic_exp = json.load(open(os.path.join(PATH_TO_CONF, "exp.conf"), "r"))
        dic_exp["MODEL_NAME"] = model_name
        dic_exp["TRAFFIC_FILE"] = traffic_file
        dic_exp["TRAFFIC_FILE_PRETRAIN"] = traffic_file_pretrain
        dic_exp["WARMUP_RUN_COUNTS"] = 3600
        dic_exp["RUN_COUNTS"] = 3600
        json.dump(dic_exp, open(os.path.join(PATH_TO_CONF, "exp.conf"), "w"), indent=4)

        dic_deeplight = json.load(open(os.path.join(PATH_TO_CONF, "deeplight_agent.conf"), "r"))
        dic_deeplight["EPSILON"] = 0.3
        json.dump(dic_deeplight, open(os.path.join(PATH_TO_CONF, "deeplight_agent.conf"), "w"), indent=4)

        # change MIN_ACTION_TIME correspondingly

        dic_cityflow = json.load(open(os.path.join(PATH_TO_CONF, "cityflow_agent.conf"), "r"))
        if model_name == "Deeplight":
            dic_cityflow["MIN_ACTION_TIME"] = 5
        else:
            dic_cityflow["MIN_ACTION_TIME"] = 1
        json.dump(dic_cityflow, open(os.path.join(PATH_TO_CONF, "cityflow_agent.conf"), "w"), indent=4)

        prefix = "{0}_{1}".format(
            dic_exp["TRAFFIC_FILE"],
            time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
        )

        traffic_light_dqn.main(memo=setting_memo, f_prefix=prefix)

        print("finished {0}".format(traffic_file))
    print("finished {0}".format(model_name))





