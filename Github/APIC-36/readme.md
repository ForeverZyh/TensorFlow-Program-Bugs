Reference: https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1

## Compatibility
The code is tested using Tensorflow r1.7 under Ubuntu 14.04 with Python 2.7 and Python 3.5. The test cases can be found [here](https://github.com/davidsandberg/facenet/tree/master/test) and the results can be found [here](http://travis-ci.org/davidsandberg/facenet).

## Set the python paths
```shell
export PYTHONPATH=[...]/facenet-*/src
```

## Training data
The [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset has been used for training. This training set consists of total of 453 453 images over 10 575 identities after face detection. Some performance improvement has been seen if the dataset has been filtered before training. Some more information about how this was done will come later.
The best performing model has been trained on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset consisting of ~3.3M faces and ~9000 classes.

## Pre-processing

```shell
python src/align/align_dataset_mtcnn.py \
  ~/datasets/casia/CASIA-maxpy-clean/ \
  ~/datasets/casia/casia_maxpy_mtcnnpy_182 \
  --image_size 182 \
  --margin 44
```

## Running training
```shell
python src/train_softmax.py \
  --logs_base_dir ~/logs/facenet/ \
  --models_base_dir ~/models/facenet/ \
  --data_dir ~/datasets/casia/casia_maxpy_mtcnnalign_182_160/ \
  --image_size 160 \
  --model_def models.inception_resnet_v1 \
  --lfw_dir ~/datasets/lfw/lfw_mtcnnalign_160/ \
  --optimizer ADAM \
  --learning_rate -1 \
  --max_nrof_epochs 150 \
  --keep_probability 0.8 \
  --random_crop \
  --random_flip \
  --use_fixed_image_standardization \
  --learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
  --weight_decay 5e-4 \
  --embedding_size 512 \
  --lfw_distance_metric 1 \
  --lfw_use_flipped_images \
  --lfw_subtract_mean \
  --validation_set_split_ratio 0.05 \
  --validate_every_n_epochs 5 \
  --prelogits_norm_loss_factor 5e-4
```
