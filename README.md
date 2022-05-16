# j-PPO+ConvNTM
This work "Energy-Efficient UAV Crowdsensing with Multiple Charging Stations by Deep Learning" has been published in INFOCOM 2020.
## :page_facing_up: Description
We aim to propose a new deep learning based framework to tackle the problem that a group of UAVs energy-efficiently and cooperatively collect data from low-level sensors, while charging the battery from multiple randomly deployed charging stations. Specifically, we propose a new deep model called "j-PPO+ConvNTM" which contains a novel spatiotemporal module "Convolution Neural Turing Machine" (ConvNTM) to better model long-sequence spatiotemporal data, and a deep reinforcement learning (DRL) model called "j-PPO", where it has the capability to make continuous (i.e., route planing) and discrete (i.e., either to collect data or go for charging) action decisions simultaneously for all UAVs. 
## :wrench: Dependencies
- Python == 3.5 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [Pytorch == 0.4.0](https://pytorch.org/)
- NVIDIA GPU (NVIDIA GTX TITAN XP) + [CUDA 10](https://developer.nvidia.com/cuda-downloads)
### Installation
1. Clone repo
    ```bash
    git clone https://github.com/BIT-MCS/jPPO_ConvLSTM.git
    cd jPPO_ConvLSTM
    ```
2. Install dependent packages
    ```
    pip install -r requirements.txt
    ```
## :zap: Quick Inference

Get the usage information of the project
```bash
cd code/uav2_charge1/exper_ppo_convmap/
python main.py -h
```
Then the usage information will be shown as following
```
usage: train.py [-h] lr num-processes DATA_WIDTH

positional arguments:
  lr  learning rate (default: 2.5e-4)
  num-processes sequence length, used for recurrent generator (default: 8)
  gamma         discount factor for rewards (default: 0.99)
 
optional arguments:
  -h, --help   show this help message and exit
```

## :computer: Training

We provide complete training codes for jPPO_ConvNTM.<br>
You could adapt it to your own needs.

1. You can modify the config files 
[jPPO_ConvNTM/code/uav2_charge1/exper_ppo_convmap/arguments.py](https://github.com/BIT-MCS/jPPO-ConvNTM/blob/main/code/uav2_charge1/exper_ppo_convmap/arguments.py) 
For example, you can set the learning rate by modifying these lines
	```
    [13] parser.add_argument('--lr', type=float, default=2.5e-4, 
    [14] help='learning rate (default: 2.5e-4)')
	```
1. Training

	```
	python main.py 
	```

## :checkered_flag: Testing
1. Before testing, you should modify the file [jPPO_ConvNTM/code/uav2_charge1/exper_ppo_convmap/test.py](https://github.com/BIT-MCS/jPPO-ConvNTM/blob/main/code/uav2_charge1/exper_ppo_convmap/test.py) as:
	```
    [18]    'MODEL_PATH' = 'your_model_saving_path'
	```
2. Testing
	```
	python test.py
	```
## :scroll: Acknowledgement

This work is supported by the National Natural Science Foundation of China (No. 61772072).

Corresponding author: Chi Harold Liu.

## :e-mail: Contact

If you have any question, please email `363317018@qq.com`.
## Paper
If you are interested in our work, please cite our paper as

```
@INPROCEEDINGS{9155535,
  author={Liu, Chi Harold and Piao, Chengzhe and Tang, Jian},
  booktitle={IEEE INFOCOM 2020 - IEEE Conference on Computer Communications}, 
  title={Energy-Efficient UAV Crowdsensing with Multiple Charging Stations by Deep Learning}, 
  year={2020},
  pages={199-208},
  doi={10.1109/INFOCOM41043.2020.9155535}
}
```
