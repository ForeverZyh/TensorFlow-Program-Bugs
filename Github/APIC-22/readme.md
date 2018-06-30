Note: A truncated version of the [original readme](https://github.com/tensorflow/models/blob/master/research/im2txt/README.md)

# Show and Tell: A Neural Image Caption Generator

## Getting Started

### A Note on Hardware and Training Time

The time required to train the *Show and Tell* model depends on your specific
hardware and computational capacity. In this guide we assume you will be running
training on a single machine with a GPU. In our experience on an NVIDIA Tesla
K20m GPU the initial training phase takes 1-2 weeks. The second training phase
may take several additional weeks to achieve peak performance (but you can stop
this phase early and still get reasonable results).

It is possible to achieve a speed-up by implementing distributed training across
a cluster of machines with GPUs, but that is not covered in this guide.

Whilst it is possible to run this code on a CPU, beware that this may be
approximately 10 times slower.

### Install Required Packages
First ensure that you have installed the following required packages:

* **Bazel** ([instructions](http://bazel.io/docs/install.html))
* **TensorFlow** 1.0 or greater ([instructions](https://www.tensorflow.org/install/))
* **NumPy** ([instructions](http://www.scipy.org/install.html))
* **Natural Language Toolkit (NLTK)**:
    * First install NLTK ([instructions](http://www.nltk.org/install.html))
    * Then install the NLTK data package "punkt" ([instructions](http://www.nltk.org/data.html))
* **Unzip**

### Prepare the Training Data

We have provided a script to download and preprocess the [MSCOCO](http://mscoco.org/) image captioning data set into this format. Downloading
and preprocessing the data may take several hours depending on your network and
computer speed. Please be patient.

Before running the script, ensure that your hard disk has at least 150GB of
available space for storing the downloaded and processed data.

```shell
# Location to save the MSCOCO data.
MSCOCO_DIR="${HOME}/im2txt/data/mscoco"

# Build the preprocessing script.
cd research/im2txt
bazel build //im2txt:download_and_preprocess_mscoco

# Run the preprocessing script.
bazel-bin/im2txt/download_and_preprocess_mscoco "${MSCOCO_DIR}"
```

The final line of the output should read:

```
2016-09-01 16:47:47.296630: Finished processing all 20267 image-caption pairs in data set 'test'.
```

When the script finishes you will find 256 training, 4 validation and 8 testing
files in `DATA_DIR`. The files will match the patterns `train-?????-of-00256`,
`val-?????-of-00004` and `test-?????-of-00008`, respectively.

### Download the Inception v3 Checkpoint

The *Show and Tell* model requires a pretrained *Inception v3* checkpoint file
to initialize the parameters of its image encoder submodel.

This checkpoint file is provided by the
[TensorFlow-Slim image classification library](https://github.com/tensorflow/models/tree/master/research/slim#tensorflow-slim-image-classification-library)
which provides a suite of pre-trained image classification models. You can read
more about the models provided by the library
[here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).


Run the following commands to download the *Inception v3* checkpoint.

```shell
# Location to save the Inception v3 checkpoint.
INCEPTION_DIR="${HOME}/im2txt/data"
mkdir -p ${INCEPTION_DIR}

wget "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
tar -xvf "inception_v3_2016_08_28.tar.gz" -C ${INCEPTION_DIR}
rm "inception_v3_2016_08_28.tar.gz"
```

## Training a Model

### Initial Training

Run the training script.

```shell
# Directory containing preprocessed MSCOCO data.
MSCOCO_DIR="${HOME}/im2txt/data/mscoco"

# Inception v3 checkpoint file.
INCEPTION_CHECKPOINT="${HOME}/im2txt/data/inception_v3.ckpt"

# Directory to save the model.
MODEL_DIR="${HOME}/im2txt/model"

# Build the model.
cd research/im2txt
bazel build -c opt //im2txt/...

# Run the training script.
bazel-bin/im2txt/train \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=1000000
```

Note: The following evaluation part is not necessary in this test.
