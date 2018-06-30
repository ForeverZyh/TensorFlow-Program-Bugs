Note: A truncated and modified version of [inception readme](https://github.com/tensorflow/models/tree/master/research/inception). This program can also use ImageNet as the dataset. For more information, see the original readme.

## Fetch the Flowers Dataset

Much like training the ImageNet model we must first convert a new data set to
the sharded TFRecord format which each entry is a serialized `tf.Example` proto.

```shell

# build the preprocessing script.
cd models-fix/inception # or models-buggy
bazel build //inception:download_and_preprocess_flowers

# run it
bazel-bin/inception/download_and_preprocess_flowers /tmp/flowers_data
```

If the script runs successfully, the final line of the terminal output should
look like:

```shell
2016-02-24 20:42:25.067551: Finished writing all 3170 images in data set.
```

## Train from Scratch

```shell
# Build the model. Note that we need to make sure the TensorFlow is ready to
# use before this as this command will not build TensorFlow.
cd models-fix # or models-buggy
bazel build //inception:flowers_train

# run it
bazel-bin/inception/flowers_train --num_gpus=1 --batch_size=32 --train_dir=/tmp/flowers_train --data_dir=/tmp/flowers_data
```
