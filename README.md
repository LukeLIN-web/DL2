# DL2
DL2 is a deep learning-driven scheduler for elastic training in deep learning clusters. DL2 advocates a joint supervised learning and reinforcement learning approach: a neural network is warmed up via offline supervised learning based on job traces produced by the existing cluster scheduler; then the neural network is plugged into the live DL cluster, fine-tuned by reinforcement learning carried out throughout the training progress of the DL jobs, and used for deciding job resource allocation in an online fashion.

Check [this figure](./workflow.pdf) for the overall workflow illustration.


## Prerequisites
We use TensorFlow to train a model. Make sure you have have installed a 1.x version:

```shell
pip install tensorflow-gpu==1.13.1
```

reference [Build from source  | TensorFlow (google.cn)](https://tensorflow.google.cn/install/source?hl=en)	

```
	python2.7, 3.3-3.7	
	GCC 4.8	
	Bazel 0.19.2	
	cuDNN7.4
	CUDA10.0
```

## Training

To train model, run the following command. It will start multiple processes to train a centralized model. 

```shell
python train.py
```

Check [parameters.py](./parameters.py) if you want to change some hyper-parameters. For ease of comparison, we also provide a script [experiment.py](./experiment.py) and you can choose different configurations.

We improve the basic policy gradient-based RL with the actor-critic algorithm, for faster convergence of the policy network.

## Trace
We put some traces collected from our testbed in [config_speed.txt](./config_speed.txt). You may need to collect your own trace if running on a different setup. For k8s setup, please check [Optimus](https://github.com/pengyanghua/optimus).

#### How do we simulate the environment of real cluster?

input:    resource allocation

ouput:  speed.

we gain func speed = fuc(num ps ,num worker) in speed.py

trace 

## Elastic Scaling

Please check the [MXNet repo](https://github.com/pengyanghua/mxnet) for the implementation of elastic resource scaling. We have modified the communication library including KVStore and pslite.

## A3C

We implement a3c in our codes.

We use a `net_gradients_qs = [multiprocessing.Queue(1) for i in range(pm.NUM_AGENTS)]`  to pass our gradients.

Each agent sends gradients to the central agent `net_gradients_q.put(policy_grads)`

Central agent polls and updates parameters,only calculate gradients once one queue is not empty.

We don't define agent is a separate class. 

```python
					if net_gradients_qs[i].qsize() == 1:
						updated_agents.append(i)
						if pm.VALUE_NET:
							policy_gradients, value_gradients = net_gradients_qs[i].get()
							value_net.apply_gradients(value_gradients)
							assert len(value_weights) == len(value_gradients)
						else:
							policy_gradients = net_gradients_qs[i].get() # without critic 
```

 

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
      * start central agent as master.  start each sl or rl agent which will send gradients to the central agent.
   
* rl_env.py
   * step(policy NN predict result)  -> masked_output, action_vec, reward, move_on, valid_state
   * 
   * 
   
* trace.py
   * Trace(policy NN predict result)  -> masked_output, action_vec, reward, move_on, valid_state
   
   * get_trace -> trace, a list which contains joblist
   
   * trace read config_speed.txt
   
   * store fitting functions in speed_funcs[model] 
   
   * 
   
   * in speed, we could get.
   
     ```
     model = resnet-50 
     sync_mode = dist_sync,
     tot_batch_size = 32, 
     num_ps = 1, 
     num_worker =1,
     speeds = list [15.783], 
     ps_cpu_usages = 1.908,
     worker_cpu_usages = 0.547
     ```
   
* drf_env.py
   * overwrite the scheduling algorithm in Scheduler,calculate the schedule time.
   * 
   
* comparison.py

   *   


## Publication
A Deep Learning-driven Scheduler for Deep Learning Clusters  arXiv:1909.06040v1 [cs.LG] 13 Sep 2019



## Remaining Problem

1. `pm.JOB_ARRIVAL_PATTERN == "Ali_Trace"` but `self.ali_trace_arrv_pattern = []` in trace.py.  Empty list without  any assignment

2. ` prob_sum = np.sum(*self*.ali_trace_job_probs[:*num_type*])` `cumsum = np.cumsum(*self*.ali_trace_job_probs[:*num_type*])`  
 `ali_trace_job_probs` is not define in any other places.

3. It seems that train starts tensorboard without send data.

  

  

  

