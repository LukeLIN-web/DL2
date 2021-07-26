# DL2
DL2 is a deep learning-driven scheduler for elastic training in deep learning clusters. DL2 advocates a joint supervised learning and reinforcement learning approach: a neural network is warmed up via offline supervised learning based on job traces produced by the existing cluster scheduler; then the neural network is plugged into the live DL cluster, fine-tuned by reinforcement learning carried out throughout the training progress of the DL jobs, and used for deciding job resource allocation in an online fashion.

Check [this figure](./workflow.pdf) for the overall workflow illustration.


## Prerequisites
We use TensorFlow to train a model. Make sure you have have installed a 1.x version:

```shell
pip install tensorflow-gpu==1.13.1
```

## Training
To train model, run the following command. It will start multiple processes to train a centralized model. 

```shell
python train.py
```

Check [parameters.py](./parameters.py) if you want to change some hyper-parameters. For ease of comparison, we also provide a script [experiment.py](./experiment.py) and you can choose different configurations.

You can also run the following command. We improve the basic policy gradient-based RL with the actor-critic algorithm, for faster convergence of the policy network.

```shell
python train_a3c.py 
```

## Trace
We put some traces collected from our testbed in [config_speed.txt](./config_speed.txt). You may need to collect your own trace if running on a different setup. For k8s setup, please check [Optimus](https://github.com/pengyanghua/optimus).

## Elastic Scaling

Please check the [MXNet repo](https://github.com/pengyanghua/mxnet) for the implementation of elastic resource scaling. We have modified the communication library including KVStore and pslite.

## Function architecture

* Train.py
   * sl_agent
      * use policy network defined at network.py
      * generate training traces
      * prepare a training batch,pull latest weights before training,superversed learning to calculate gradients,send gradients to the central agent, validation
   * rl_agent
      *  select use experience replay or not, using according replay buffer. 
      * generate training data
      * select use epsilon greedy or not, using according temperature. 
      * send gradients to the central agent
      *  validation
      * collect statistics after training one trace
   * main
      * start central agent, start each sl or rl agents
* rl_env.py
   * step(policy NN predict result)  -> masked_output, action_vec, reward, move_on, valid_state
   * 
   * 
* trace.py
   * Trace(policy NN predict result)  -> masked_output, action_vec, reward, move_on, valid_state
   * get_trace(numtype=8) -> trace, a list which contains joblist
   *  

## Publication
A Deep Learning-driven Scheduler for Deep Learning Clusters  arXiv:1909.06040v1 [cs.LG] 13 Sep 2019
