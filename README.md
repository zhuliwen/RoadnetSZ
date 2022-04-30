# RoadnetSZ: Road networks and traffic flows in Shenzhen

![GitHub last commit](https://img.shields.io/github/last-commit/zhuliwen/RoadnetSZ) ![issues](https://img.shields.io/github/issues/zhuliwen/RoadnetSZ)

In this repository, we release the **Shenzhen** dataset and code for multi-agent traffic signal control.

- **Dataset**. We provide two versions that can run on both [SUMO](http://sumo.dlr.de/index.html) and [CityFlow](https://github.com/cityflow-project/CityFlow) platforms.
- **Code**. We provide three RL-based methods as baselines.

## Table of contents

- [RoadnetSZ](#RoadnetSZ)
	- [Table of contents](#table-of-contents)
	- [Open Datasets](#open-datasets)
	- [Code list](#code-list)
	- [Cite](#cite)

## Open Datasets

| #    | Platform | Figure                      | Dataset                                                      |
| ---- | -------- | --------------------------- | ------------------------------------------------------------ |
| 1    | CityFlow | ![1](./img/fuhua2.png)      | **Roadnet**:<br> [roadnet_1_33.json](./data_cityflow/roadnet_1_33.json) <br>**Flow**:<br>[anon_1_33_fuhua_24hto1w_2490.json](./data_cityflow/anon_1_33_fuhua_24hto1w_2490.json)<br>[anon_1_33_fuhua_4_27_24hto1w_4089.json](./data_cityflow/anon_1_33_fuhua_4_27_24hto1w_4089.json)<br> |
| 2    | CityFlow | ![1](./img/fuhua.JPEG)      | **Roadnet**:<br> [fuhua_cityflow.json](./data_cityflow/fuhua_cityflow.json) <br>**Flow**:<br>[fuhua_real_1775.json](./data_cityflow/fuhua_real_1775.json)<br>[fuhua_2570.json](./data_cityflow/fuhua_2570.json)<br>[fuhua_4770.json](./data_cityflow/fuhua_4770.json) |
| 3    | SUMO     | ![2](./img/futian_sumo.jpg) | [FuTian.net.xml](./data_sumo/FuTian.net.xml)<br>[FuTian.edg.xml](./data_sumo/FuTian.edg.xml)<br>[FuTian.nod.xml](./data_sumo/FuTian.nod.xml)<br>[FuTian.tll.xml](./data_sumo/FuTian.tll.xml)<br>[FuTian.typ.xml](./data_sumo/FuTian.typ.xml)<br>[FuTian.con.xml](./data_sumo/FuTian.con.xml)<br> |
| 4    | SUMO     | ![3](./img/baoan_sumo.jpg)  | [BaoAn.net.xml](./data_sumo/BaoAn.net.xml)<br>[BaoAn.edg.xml](./data_sumo/BaoAn.edg.xml)<br>[BaoAn.nod.xml](./data_sumo/BaoAn.nod.xml)<br>[BaoAn.tll.xml](./data_sumo/BaoAn.tll.xml)<br>[BaoAn.typ.xml](./data_sumo/BaoAn.typ.xml)<br>[BaoAn.con.xml](./data_sumo/BaoAn.con.xml)<br> |

## Code list

| #    | Title                                                        | Figure                  | Code                   | Tutorial                             |
| ---- | ------------------------------------------------------------ | ----------------------- | ---------------------- | ------------------------------------ |
| 1    | [Hierarchically and Cooperatively Learning Traffic Signal Control](https://z0ngqing.github.io/paper/aaai-bingyu21.pdf) | ![1](./img/hilight.png) | [code](./code/HiLight) | [tutorial](./code/HiLight/README.md) |
| 2    | [MetaVIM: Meta Variationally Intrinsic Motivated Reinforcement Learning for Decentralized Traffic Signal Control](https://arxiv.org/pdf/2101.00746.pdf) | ![2](./img/metavim.png) | [code](./code/MetaVIM) | [tutorial](./code/MetaVIM/README.md)  |
| 3    | [Learning To Share In Multi-Agent Reinforcement Learning](https://openreview.net/pdf?id=awnQ2qTLSwn) | ![3](./img/ltos.png) | [code](./code/LToS)    | [tutorial](./code/LToS/README.md)    |




## Cite

If you use Shenzhen Dataset in your work, please cite it as follows:

```
@misc{RoadnetSZ,
	title = {RoadnetSZ},
	author = {Bingyu, Xu and Liwen, Zhu and Yuxuan, Yi and Zongqing, Lu and other contributors},
	year = {2022},
	howpublished = {\url{https://github.com/zhuliwen/PKU_Traffic_Lights}},
	note = {Accessed: 2022-05-01},
}

@inproceedings{xu2021hierarchically,
  title={Hierarchically and cooperatively learning traffic signal control},
  author={Xu, Bingyu and Wang, Yaowei and Wang, Zhaozhi and Jia, Huizhu and Lu, Zongqing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={1},
  pages={669--677},
  year={2021}
}

@article{zhu2021variationally,
  title={Variationally and intrinsically motivated reinforcement learning for decentralized traffic signal control},
  author={Zhu, Liwen and Peng, Peixi and Lu, Zongqing and Wang, Xiangqian and Tian, Yonghong},
  journal={arXiv preprint arXiv:2101.00746},
  year={2021}
}

@article{yi2021learning,
  title={Learning to Share in Multi-Agent Reinforcement Learning},
  author={Yi, Yuxuan and Li, Ge and Wang, Yaowei and Lu, Zongqing},
  journal={arXiv preprint arXiv:2112.08702},
  year={2021}
}
```
