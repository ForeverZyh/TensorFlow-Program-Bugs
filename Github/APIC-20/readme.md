Note: This is a truncated version of the [original readme](https://github.com/tensorflow/models/tree/master/research/slim/README.md)

# TensorFlow-Slim image classification model library

## Contacts

Maintainers of TF-slim:

* Nathan Silberman,
  github: [nathansilberman](https://github.com/nathansilberman)
* Sergio Guadarrama, github: [sguada](https://github.com/sguada)

# Preparing the datasets

## Downloading and converting to TFRecord format

```shell
$ DATA_DIR=/tmp/flowers
$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"
```

When the script finishes you will find several TFRecord files created:

```shell
$ ls ${DATA_DIR}
flowers_train-00000-of-00005.tfrecord
...
flowers_train-00004-of-00005.tfrecord
flowers_validation-00000-of-00005.tfrecord
...
flowers_validation-00004-of-00005.tfrecord
labels.txt
```

These represent the training and validation data, sharded over 5 files each.
You will also find the `$DATA_DIR/labels.txt` file which contains the mapping
from integer labels to class names.

You can use the same script to create the mnist and cifar10 datasets.
However, for ImageNet, you have to follow the instructions
[here](https://github.com/tensorflow/models/blob/master/research/inception/README.md#getting-started).
Note that you first have to sign up for an account at image-net.org.
Also, the download can take several hours, and could use up to 500GB.

# Pre-trained Models

Here is an example of how to download the Inception-ResNet-v2 checkpoint:

```shell
$ CHECKPOINT_DIR=/tmp/checkpoints
$ mkdir ${CHECKPOINT_DIR}
$ wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
$ tar -xvf vgg_16_2016_08_28.tar.gz
$ mv vgg_16.ckpt ${CHECKPOINT_DIR}
$ rm vgg_16_2016_08_28.tar.gz
```

# Fine-tuning a model from an existing checkpoint

```shell
$ DATASET_DIR=/tmp/flowers
$ TRAIN_DIR=/tmp/flowers-models/vgg_16
$ CHECKPOINT_PATH=/tmp/my_checkpoints/vgg_16.ckpt
$ python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --model_name=vgg_16 \
    --checkpoint_path=${CHECKPOINT_PATH} \
```
