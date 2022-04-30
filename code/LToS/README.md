# Learning to Share in Multi-Agent Reinforcement Learning

This project LToS has been published as the following conference paper in ICLR workshop:

```
@inproceedings{ltos,
  author    = {Yuxuan Yi and
               Ge Li and
               Yaowei Wang and
               Zongqing Lu},
  title     = {Learning to Share in Multi-Agent Reinforcement Learning},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR) workshop},
  series = {ICLR workshop '22},
  pages     = {1-16},
  year      = {2022},
}
```

# Introduction
A Traffic signal control based on multi-agent reinforcement learning with a newly-proposed hierarchical framework: SWARM. *(In our paper we rename it as LToS)*

## 1. Function
This code is modified on the basis of https://github.com/wingsweihua/colight . In contrast, we focus on the cooperative control problem between multiple traffic lights in this project, and apply a newly-proposed framework: SWARM for global optimization.

Here we provide the map we use in the paper: a 6x6 grid map whose block is 100 meters long and 100 meters wide. Modify the road net in the following code of runexp.py . So please just let the following arguments be.

    parser.add_argument("--memo", type=str, default='0515_afternoon_Colight_6_6_bi') # meaningless default name
    parser.add_argument("--road_net", type=str, default='6_6') # or '1_33'
    parser.add_argument("--volume", type=str, default='small') # or 'fuhua'
    parser.add_argument("--suffix", type=str, default="2570_bi") # or '0.4x24hto1h'

And you may modify the following arguments in the function parse_args:

    NUM_ROUNDS = 100 # training round: 100 for small_6_6; 400 for fuhua_1_33
    parser.add_argument("--mod", type=str, default="SwarmCoLight") # model name;
    
    parser.add_argument("--cnt", type=int, default=3600) # episode length
    # You may try other length, but remember as the original traffic flow file confines, there will be no more vehicles after 3600s.

    parser.add_argument("--gen", type=int, default=4) # generator: 4 for small_6_6, 1 for fuhua_1_33
    # We test once every time we finish #gen times of training. Therefore, actually we train for 100*4=400*1=400 times now.

Caution: In case of collision with Django, please don't use any arguments in terminal and leave all arguments in parse_args() be as default.

## 2. Efficiency
We have an alteration about platform from SUMO and its python interface for CityFlow (https://cityflow-project.github.io/), which shows much higher efficiency when there are many intersections and vehicles. We train and test our model on the basis of 100 episodes of 3600-second-long simulations. 

Please follow the instructions of https://cityflow-project.github.io/ to install CityFlow.

Caution: Large amount of other statistics will also be outputted to stdout and in the folder records/ and summary/, and thus you'd better run with nohup command (both here and below).

For training, run:
    
    python3 runexp.py
    
### (a) small_6_6:

It will lead to a 100-round-long iteration (NUM_ROUNDS: number of rounds can be modified) where every round contains one training episode (with network updated) and one testing episode (without network updated). It will run faster on a multi-core server with GPUs. When NUM_ROUNDS=100, gen=4 and cnt=3600, We give one evaluation instance which was run with one Tesla M10 GPU:
    
    1 whole training round: about 350s
    1 testing round: about 50s
    1 round: about 400s; all 100 rounds: about 40000s = 11.11h

You can see your efficiency in the folder records/(your instance name)/(anon_your instance name_your instance time)/running_time.csv like this:
    
    generator_time  making_samples_time     update_network_time     test_evaluation_times   all_times
    254.6277620792389      34.41663098335266      48.11376166343689      63.60378074645996       400.76217126846313
    ...
    
### (b) fuhua_1_33 + 0.4x24hto1h.json:

**Roadnet**:<br> [fuhua_cityflow.json](./data_cityflow/fuhua_cityflow.json) <br>**Flow**:<br>[fuhua_real_1775.json](./data_cityflow/fuhua_real_1775.json). (NUM_ROUNDS: number of rounds can be modified) where every round contains one training episode (with network updated) and one testing episode (without network updated). It will run faster on a multi-core server with GPUs. When NUM_ROUNDS=400, gen=1 and cnt=3600, We give one evaluation instance which was run with one Tesla M10 GPU:
    
    1 whole training round: about 80s
    1 testing round: about 45s
    1 round: about 125s; all 400 rounds: about 50000s = 13.89h

You can see your efficiency in the folder records/(your instance name)/(anon_your instance name_your instance time)/running_time.csv like this:
    
    generator_time  making_samples_time     update_network_time     test_evaluation_times   all_times
    51.30196952819824       7.750897169113159       19.962382078170776      45.88180637359619       124.89742064476013
    ...

And get your model in the folder model/(your instance name)/(anon_your instance name_your instance time), records in the folder records/(your instance name)/(anon_your instance name_your instance time). And you may get your summary/(your instance name)/(anon_your instance name_your instance time) after running:

    python3 summary_multi_anon.py
 
## 3. Metrics
A common goal is to minimize average duration (travel time). To achieve that, previous works tended to use multiple metrics like wait time, queue length, delay and so on. We make some alterations to the state representation and reward setting. We choose queue length as reward, which is proved effective enough for traffic optimization by model construction.

At the end of each episode, some statistics like the follows will be given in the folder summary/(your instance name)/(anon_your instance name_your instance time)/test_results.csv like this: 

(a) small_6_6

    ,duration,queue_length,vehicle_in,vehicle_out
    0,2217.62893081761,15.555555555555555,954,662
    ...
    99,98.59533073929961,0.0,2570,2564

(b) fuhua_1_33 + 0.4x24hto1h.json

    ,duration,queue_length,vehicle_in,vehicle_out
    0,1294.5055643879173,12.878787878787879,1258,846
    ...
    346,203.85915492957747,0.5454545454545454,1775,1761
    ...

The time data (in seconds) reflect how many vehicles have left the network (the more, the better) and how long in average they take to finish their journey (the less, the better).

## 4. Dataset

### Vehicle information:
    data/(road net name)/(number of rows and cols)/anon_(number of rows and cols)_(road net name)_(volume).json
### Net information:
    data/(road net name)/(number of rows and cols)/roadnet_(number of rows and cols).json
Every road can be uni/bi-directional. Every traffic light has 8 phases to choose (4 of them are yellow-light phases).

Check config.py and runexp.py for method setting like hyperparameters.

# Environment & Dependency:
|Type|Name|Version|
|---|---|---|
|language|python|>=3.5|
|simulation platform|CityFlow|>=1.0.0|
|frame|keras||
|package|h5py||
|backend|Tensorflow/Tensorflow-gpu||
|package|matplotlib||
|package|numpy||
|package|pandas||

install CUDA & cuDNN, if you use a gpu to train the model.

My configure:

|Type|Name|Version|
|---|---|---|
|os|Ubuntu|16.04.6 LTS (GNU/Linux 4.4.0-190-generic x86_64)|
|language|python|3.6|
|simulation platform|CityFlow|1.1.0|
|package|matplotlib|3.0.3|
|package|numpy|1.16.2|
|package|pandas|0.24.2|
|package|h5py|2.9.0|
|frame|Keras|2.2.4|
|backend|tensorflow-gpu|1.13.1|
|gpu|CUDA|10.1.243|
|gpu|CUDNN|7.4.2|

# Input & Output
|Name|Description|
|---|---|
|Input|episode length; other arguments are offered in the form of files.|
|Output|pandas.dataframe which contains the following performance: {average duration (travel time); queue length (average waiting queue length); vehicle_in (number of vehicles that managed to come in), vehicle_out (number of vehicles that managed to go out)}|
|Statistics|large amount of other statistics will also be provided to stdout and in the folder records/ and summary/, and thus you'd better run with nohup command.|

## Model and records:
The model will be created in the subdirectorys in the model/ directory.

The records will be created in the subdirectorys in the records/ directory.

Every subdirectory will be named in this format:
    
    (your instance name)/(anon_your instance name_your instance time)

e.g.
    
    model/0515_afternoon_Colight_6_6_bi/anon_6_6_300_0.3_bi.json_05_21_22_24_02

# How to run
In terminal:
```shell
cd project_dir
```
and then:
```shell
python3 runexp.py
python3 summary_multi_anon.py
```
