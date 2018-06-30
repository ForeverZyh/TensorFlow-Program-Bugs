Note: This is a truncated version of the [original readme](https://github.com/tensorflow/models/blob/master/research/video_prediction/README.md)

# Video Prediction with Neural Advection

*A TensorFlow implementation of the models described in [Unsupervised Learning for Physical Interaction through Video Prediction (Finn et al., 2016)](https://arxiv.org/abs/1605.07157).*

## Requirements
* Tensorflow (see tensorflow.org for installation instructions)
* spatial_tranformer model in tensorflow/models, for the spatial tranformer
  predictor (STP).

## Data
The data used to train this model is located
[here](https://sites.google.com/site/brainrobotdata/home/push-dataset).

To download the robot data, run the following.
```shell
./download_data.sh
```

## Training the model

To train the model, run the prediction_train.py file.
```shell
python prediction_train.py
```
