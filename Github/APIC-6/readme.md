**NOTE: For the most part, you will find a newer version of this code at [models/research/slim](https://github.com/tensorflow/models/tree/master/research/slim).** In particular:

*   `inception_train.py` and `imagenet_train.py` should no longer be used. The slim editions for running on multiple GPUs are the current best examples.
*   `inception_distributed_train.py` and `imagenet_distributed_train.py` are still valid examples of distributed training.

For performance benchmarking, please see https://www.tensorflow.org/performance/benchmarks.

---

# Inception in TensorFlow

[ImageNet](http://www.image-net.org/) is a common academic data set in machine
learning for training an image recognition system. Code in this directory
demonstrates how to use TensorFlow to train and evaluate a type of convolutional
neural network (CNN) on this academic data set. In particular, we demonstrate
how to train the Inception v3 architecture as specified in:

_Rethinking the Inception Architecture for Computer Vision_

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew
Wojna

http://arxiv.org/abs/1512.00567

This network achieves 21.2% top-1 and 5.6% top-5 error for single frame
evaluation with a computational cost of 5 billion multiply-adds per inference
and with using less than 25 million parameters. Below is a visualization of the
model architecture.

![Inception-v3 Architecture](g3doc/inception_v3_architecture.png)

## Description of Code

The code base provides three core binaries for:

*   Training an Inception v3 network from scratch across multiple GPUs and/or
    multiple machines using the ImageNet 2012 Challenge training data set.
*   Evaluating an Inception v3 network using the ImageNet 2012 Challenge
    validation data set.
*   Retraining an Inception v3 network on a novel task and back-propagating the
    errors to fine tune the network weights.

The training procedure employs synchronous stochastic gradient descent across
multiple GPUs. The user may specify the number of GPUs they wish to harness. The
synchronous training performs *batch-splitting* by dividing a given batch across
multiple GPUs.

The training set up is nearly identical to the section [Training a Model Using
Multiple GPU Cards](https://www.tensorflow.org/tutorials/deep_cnn/index.html#launching_and_training_the_model_on_multiple_gpu_cards)
where we have substituted the CIFAR-10 model architecture with Inception v3. The
primary differences with that setup are:

*   Calculate and update the batch-norm statistics during training so that they
    may be substituted in during evaluation.
*   Specify the model architecture using a (still experimental) higher level
    language called TensorFlow-Slim.

For more details about TensorFlow-Slim, please see the [Slim README](inception/slim/README.md). Please note that this higher-level language is still
*experimental* and the API may change over time depending on usage and
subsequent research.

## Getting Started

Before you run the training script for the first time, you will need to download
and convert the ImageNet data to native TFRecord format. The TFRecord format
consists of a set of sharded files where each entry is a serialized `tf.Example`
proto. Each `tf.Example` proto contains the ImageNet image (JPEG encoded) as
well as metadata such as label and bounding box information. See
[`parse_example_proto`](inception/image_processing.py) for details.

We provide a single [script](inception/data/download_and_preprocess_imagenet.sh) for
downloading and converting ImageNet data to TFRecord format. Downloading and
preprocessing the data may take several hours (up to half a day) depending on
your network and computer speed. Please be patient.

To begin, you will need to sign up for an account with [ImageNet](http://image-net.org) to gain access to the data. Look for the sign up page,
create an account and request an access key to download the data.

After you have `USERNAME` and `PASSWORD`, you are ready to run our script. Make
sure that your hard disk has at least 500 GB of free space for downloading and
storing the data. Here we select `DATA_DIR=$HOME/imagenet-data` as such a
location but feel free to edit accordingly.

When you run the below script, please enter *USERNAME* and *PASSWORD* when
prompted. This will occur at the very beginning. Once these values are entered,
you will not need to interact with the script again.

```shell
# location of where to place the ImageNet data
DATA_DIR=$HOME/imagenet-data

# build the preprocessing script.
cd tensorflow-models/inception
bazel build //inception:download_and_preprocess_imagenet

# run it
bazel-bin/inception/download_and_preprocess_imagenet "${DATA_DIR}"
```

The final line of the output script should read:

```shell
2016-02-17 14:30:17.287989: Finished writing all 1281167 images in data set.
```

When the script finishes, you will find 1024 training files and 128 validation
files in the `DATA_DIR`. The files will match the patterns
`train-?????-of-01024` and `validation-?????-of-00128`, respectively.

[Congratulations!](https://www.youtube.com/watch?v=9bZkp7q19f0) You are now
ready to train or evaluate with the ImageNet data set.

## How to Train from Scratch in a Distributed Setting

**NOTE** Distributed TensorFlow requires version 0.8 or later.

Distributed TensorFlow lets us use multiple machines to train a model faster.
This is quite different from the training with multiple GPU towers on a single
machine where all parameters and gradients computation are in the same place. We
coordinate the computation across multiple machines by employing a centralized
repository for parameters that maintains a unified, single copy of model
parameters. Each individual machine sends gradient updates to the centralized
parameter repository which coordinates these updates and sends back updated
parameters to the individual machines running the model training.

We term each machine that runs a copy of the training a `worker` or `replica`.
We term each machine that maintains model parameters a `ps`, short for
`parameter server`. Note that we might have more than one machine acting as a
`ps` as the model parameters may be sharded across multiple machines.

Variables may be updated with synchronous or asynchronous gradient updates. One
may construct a an [`Optimizer`](https://www.tensorflow.org/api_docs/python/train.html#optimizers) in TensorFlow
that constructs the necessary graph for either case diagrammed below from the
TensorFlow [Whitepaper](http://download.tensorflow.org/paper/whitepaper2015.pdf):

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%"
  src="https://www.tensorflow.org/images/tensorflow_figure7.png">
</div>

In [a recent paper](https://arxiv.org/abs/1604.00981), synchronous gradient
updates have demonstrated to reach higher accuracy in a shorter amount of time.
In this distributed Inception example we employ synchronous gradient updates.

Note that in this example each replica has a single tower that uses one GPU.

The command-line flags `worker_hosts` and `ps_hosts` specify available servers.
The same binary will be used for both the `worker` jobs and the `ps` jobs.
Command line flag `job_name` will be used to specify what role a task will be
playing and `task_id` will be used to identify which one of the jobs it is
running. Several things to note here:

*   The numbers of `ps` and `worker` tasks are inferred from the lists of hosts
    specified in the flags. The `task_id` should be within the range `[0,
    num_ps_tasks)` for `ps` tasks and `[0, num_worker_tasks)` for `worker`
    tasks.
*   `ps` and `worker` tasks can run on the same machine, as long as that machine
    has sufficient resources to handle both tasks. Note that the `ps` task does
    not benefit from a GPU, so it should not attempt to use one (see below).
*   Multiple `worker` tasks can run on the same machine with multiple GPUs so
    machine_A with 2 GPUs may have 2 workers while machine_B with 1 GPU just has
    1 worker.
*   The default learning rate schedule works well for a wide range of number of
    replicas [25, 50, 100] but feel free to tune it for even better results.
*   The command line of both `ps` and `worker` tasks should include the complete
    list of `ps_hosts` and `worker_hosts`.
*   There is a chief `worker` among all workers which defaults to `worker` 0.
    The chief will be in charge of initializing all the parameters, writing out
    the summaries and the checkpoint. The checkpoint and summary will be in the
    `train_dir` of the host for `worker` 0.
*   Each worker processes a batch_size number of examples but each gradient
    update is computed from all replicas. Hence, the effective batch size of
    this model is batch_size * num_workers.

```shell
# Build the model. Note that we need to make sure the TensorFlow is ready to
# use before this as this command will not build TensorFlow.
cd tensorflow-models/inception
bazel build //inception:imagenet_distributed_train

# To start worker 0, go to the worker0 host and run the following (Note that
# task_id should be in the range [0, num_worker_tasks):
bazel-bin/inception/imagenet_distributed_train \
--batch_size=32 \
--data_dir=$HOME/imagenet-data \
--job_name='worker' \
--task_id=0 \
--ps_hosts='ps0.example.com:2222' \
--worker_hosts='worker0.example.com:2222,worker1.example.com:2222'

# To start worker 1, go to the worker1 host and run the following (Note that
# task_id should be in the range [0, num_worker_tasks):
bazel-bin/inception/imagenet_distributed_train \
--batch_size=32 \
--data_dir=$HOME/imagenet-data \
--job_name='worker' \
--task_id=1 \
--ps_hosts='ps0.example.com:2222' \
--worker_hosts='worker0.example.com:2222,worker1.example.com:2222'

# To start the parameter server (ps), go to the ps host and run the following (Note
# that task_id should be in the range [0, num_ps_tasks):
bazel-bin/inception/imagenet_distributed_train \
--job_name='ps' \
--task_id=0 \
--ps_hosts='ps0.example.com:2222' \
--worker_hosts='worker0.example.com:2222,worker1.example.com:2222'
```

If you have installed a GPU-compatible version of TensorFlow, the `ps` will also
try to allocate GPU memory although it is not helpful. This could potentially
crash the worker on the same machine as it has little to no GPU memory to
allocate. To avoid this, you can prepend the previous command to start `ps`
with: `CUDA_VISIBLE_DEVICES=''`

```shell
CUDA_VISIBLE_DEVICES='' bazel-bin/inception/imagenet_distributed_train \
--job_name='ps' \
--task_id=0 \
--ps_hosts='ps0.example.com:2222' \
--worker_hosts='worker0.example.com:2222,worker1.example.com:2222'
```

If you have run everything correctly, you should see a log in each `worker` job
that looks like the following. Note the training speed varies depending on your
hardware and the first several steps could take much longer.

```shell
INFO:tensorflow:PS hosts are: ['ps0.example.com:2222', 'ps1.example.com:2222']
INFO:tensorflow:Worker hosts are: ['worker0.example.com:2222', 'worker1.example.com:2222']
I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:206] Initialize HostPortsGrpcChannelCache for job ps -> {ps0.example.com:2222, ps1.example.com:2222}
I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:206] Initialize HostPortsGrpcChannelCache for job worker -> {localhost:2222, worker1.example.com:2222}
I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:202] Started server with target: grpc://localhost:2222
INFO:tensorflow:Created variable global_step:0 with shape () and init <function zeros_initializer at 0x7f6aa014b140>

...

INFO:tensorflow:Created variable logits/logits/biases:0 with shape (1001,) and init <function _initializer at 0x7f6a77f3cf50>
INFO:tensorflow:SyncReplicas enabled: replicas_to_aggregate=2; total_num_replicas=2
INFO:tensorflow:2016-04-13 01:56:26.405639 Supervisor
INFO:tensorflow:Started 2 queues for processing input data.
INFO:tensorflow:global_step/sec: 0
INFO:tensorflow:Worker 0: 2016-04-13 01:58:40.342404: step 0, loss = 12.97(0.0 examples/sec; 65.428  sec/batch)
INFO:tensorflow:global_step/sec: 0.0172907
...
```

and a log in each `ps` job that looks like the following:

```shell
INFO:tensorflow:PS hosts are: ['ps0.example.com:2222', 'ps1.example.com:2222']
INFO:tensorflow:Worker hosts are: ['worker0.example.com:2222', 'worker1.example.com:2222']
I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:206] Initialize HostPortsGrpcChannelCache for job ps -> {localhost:2222, ps1.example.com:2222}
I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:206] Initialize HostPortsGrpcChannelCache for job worker -> {worker0.example.com:2222, worker1.example.com:2222}
I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:202] Started server with target: grpc://localhost:2222
```

If you compiled TensorFlow (from v1.1-rc3) with VERBS support and you have the
required device and IB verbs SW stack, you can specify --protocol='grpc+verbs'
In order to use Verbs RDMA for Tensor passing between workers and ps.
Need to add the the --protocol flag in all tasks (ps and workers).
The default protocol is the TensorFlow default protocol of grpc.


[Congratulations!](https://www.youtube.com/watch?v=9bZkp7q19f0) You are now
training Inception in a distributed manner.
