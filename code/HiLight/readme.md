# Hierarchically and Cooperatively Learning Traffic Signal Control

**Introduction**

This project HiLight has been published as the following conference paper in AAAI:

```
@inproceedings{hilight,
  author    = {Bingyu Xu and
               Yaowei Wang and
               Zhaozhi Wang and
               Huizhu Jia and
               Zongqing Lu},
  title     = {Hierarchically and Cooperatively Learning Traffic Signal Control},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  series = {AAAI '21},
  pages     = {669--677},
  publisher = {{AAAI} Press},
  year      = {2021},
}
```

Before running above codes, you may need to install following packages or environments:

- Python 3.6
- CityFlow 0.1 (Installation Guide: https://cityflow.readthedocs.io/en/latest/install.html)
- Keras 2.3.1 

Start an experiment by:

``python -O runexp.py``

Here, ``-O`` option cannot be omitted unless debug is necessary. In the file ``runexp.py``, the args can be changed.

* ``runexp.py``

  Run the pipeline under different traffic flows. Specific traffic flow files as well as basic configuration can be assigned in this file. For details about config, please turn to ``config.py``.

For most cases, you might only modify traffic files and config parameters in ``runexp.py``.

To run different road networks, set the parameters in runexp.py, traffic_light_dqn.py, map_computer.py accordingly. 

## Dataset

* conf/

Configuration files.

* data/

Traffic flow files.

* synthetic data

  Traffic file and road networks can be found in ``data/4_4``.

* real-world data

  Traffic file and road networks can be found in ``data/jinan`` && ``data/manhattan_16_3`` && ``data/shenzhen`` (the last is our original dataset).

## Structure

Other files follow the project structure of IntelliLight: https://github.com/wingsweihua/IntelliLight

```
@inproceedings{wei2018intellilight,
  title={IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control},
  author={Wei, Hua and Zheng, Guanjie and Yao, Huaxiu and Li, Zhenhui},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2496--2505},
  year={2018},
  organization={ACM}
}
```

The files are functioning like this:

* runexp.py

Load the experiment setting, and call traffic_light_agent.

* traffic_light_dqn.py

Simulate the reinforcement learning agent. This file simulates the timeline, get current state from sumo_agent, get current state from sumo_agent and call the agent to make decision.

* sumo_agent.py

Interact with map_computor to get state, reward from the map_computor, and convey action to map_computor.

* map_computor.py

Read data from CityFlow and operate CityFlow.

* agent.py

Abstract class of agent.

* network_agent.py

Abstract class of neural network based agent.

* deeplight_agent.py

Class of our method.

## Baseline

CoLight: https://github.com/wingsweihua/colight

```
@inproceedings{colight,
 author = {Wei, Hua and Xu, Nan and Zhang, Huichu and Zheng, Guanjie and Zang, Xinshi and Chen, Chacha and Zhang, Weinan and Zhu, Yamin and Xu, Kai and Li, Zhenhui},
 title = {CoLight: Learning Network-level Cooperation for Traffic Signal Control},
 booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
 series = {CIKM '19},
 year = {2019},
 location = {Beijing, China}
} 
```