# MetaVIM



**MetaVIM** (Meta Variationally Intrinsic Motivated) is a RL method, aim to learn the decentralized policies of each traffic signal only conditioned on its local observation.

Usage and more information can be found below.



## Installation

This repo is tested on Python 3.6+, CityFlow 0.1, PyTorch 1.0.0+ (PyTorch 1.5.1 for examples) and TensorFlow 1.14.0.


## Dataset

This repo containes four real-world datasets (Hangzhou, Jinan, New York and Shenzhen), which are stored in the `./data`

- Road network map

  - | File Name           | City              |
    | ------------------- | ----------------- |
    | roadnet_4_4.json    | Hangzhou          |
    | roadnet_3_4.json    | Jinan             |
    | roadnet_16_3.json   | New York          |
    | fuhua_cityflow.json | Shenzhen          |
    | roadnet_3_3.json    | grid_road_network |
    | roadnet_6_6.json    | grid_road_network |
    | roadnet_10_10.json  | grid_road_network |

- Traffic flow
  - `*raw` denotes the `real` data.
  - `*2570` denotes the `mixed_low` data.
  - `*4770` denotes the `mixed_high` data.

  

## Usage

Run the code:

1. Firstly, set the `city_and_configuration` in `main.py` according to the following format:

```python
[City]_[Pattern]
```

- City can be *hangzhou*, *jinan*, *newyork*, *shenzhen*.
- Pattern can be *real*, *mixed_low*, *mixed_high*.

2. Secondly, run the `main.py` :

Example1:

```sh
CUDA_VISIBLE_DEVICES=1 \
python main.py \
  --log_name 6_6_withneighbor_2570 \
  --road_net 6_6 \
  --volume 300 \
  --suffix 0.3_bi \
  --path_to_log records/test/6_6/train_round \
  --path_to_work_directory data \
  --num_processes 36 \
  --use_neighbor True \
  --use_intrinsic_reward True \
  --decode_state_neighbor True \
  --decode_reward_neighbor True \
  --input_action_neighbor True
```

Example2:

```shell
CUDA_VISIBLE_DEVICES=1 \
python main.py \
  --log_name hangzhou_withneighbor_2570 \
  --road_net 4_4 \
  --volume hangzhou \
  --suffix real \
  --path_to_log records/test/hangzhou/train_round \
  --path_to_work_directory data \
  --num_processes 16 \
  --use_neighbor True \
  --use_intrinsic_reward True \
  --decode_state_neighbor True \
  --decode_reward_neighbor True \
  --input_action_neighbor True
```



















