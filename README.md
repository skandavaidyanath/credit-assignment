# Hindsight-DICE: Stable Credit Assignment for Deep Reinforcement Learning
This repository contains the official code for [Hindsight-DICE: Stable Credit Assignment for Deep Reinforcement Learning](https://arxiv.org/abs/2307.11897). For full details of the H-DICE method, benchmark environmnets, baseline algorithms, experiments, and hyperparameter settings, please refer to the paper linked above. 

# Setup
Our implementation of H-DICE and baseline methods uses PyTorch. After cloning this repository, please create and activate a Python 3.9 Conda environment by following the steps [here](https://www.google.com/search?q=create+a+conda+environment&rlz=1C5GCEM_enUS1067US1067&oq=create+a+conda+environment+&aqs=chrome..69i57j0i512l2j69i59i512j0i512l6.5051j0j7&sourceid=chrome&ie=UTF-8). Then use ```conda install --file requirements.txt``` to install required packages. Note that the packages in ```requirements.txt``` may not be comprehensive, and that you may need to install additional requirements when running code. 

We use Weights and Biases (wandb) to log data for our experiments. To setup a wandb account, refer to [this](https://docs.wandb.ai/quickstart) page. Wandb logging can be disabled in experiments by setting logger.wandb=false when running experiments via command line. 

# Running Experiments
This repository implements H-DICE and three baseline methods - PPO, PPO-HCA, and PPO-HCA-Clip - and is able to evaluate these methods in GridWorld, LunarLander, and Mujoco environments. Environment settings and hyperparameters are configured using Hydra. Config files are found in ```src/conf```; environment configs are further found without the ```env/``` subdirectory while algorithm configs are found in ```agent/```. A basic experiment can be run via command line by specifying only the agent-type, a string which must match a filename (without ```.yaml```) which exists in ```agent/```, and the env, which similarly must match a filename that exists within the ```env/``` subdirectory. Configs specify default settings for hyperparameters; hyperparameters can be overriden in the command line. For instance, the following command runs an experiment in with H-DICE as the algorithm in GridWorld-v2 (which in our repository is referred to as GridWorld-v6):

```python train.py env=gridworld/gridworld_v6 agent=hca-dualdice training.seed=1 logger.exp_name_modifier=seed1 logger.group_name_modifier=v6_psi_uniform_seeds logger.wandb=true agent.value_loss_coeff=0.0 agent.psi=uniform```


Descriptions of hyperparameters can be found within the respective config files. 


# Credits
If you use this codebase for your work, please cite the paper:

```
@misc{velu2023hindsightdice,
      title={Hindsight-DICE: Stable Credit Assignment for Deep Reinforcement Learning}, 
      author={Akash Velu and Skanda Vaidyanath and Dilip Arumugam},
      year={2023},
      eprint={2307.11897},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

The PPO implementation in this codebase is inspired from [this](https://github.com/nikhilbarhate99/PPO-PyTorch) repository. 

