Reference: [Image Completion with Deep Learning in TensorFlow](http://bamos.github.io/2016/08/09/deep-completion/)

## openface (for pre-process)

Before this, install OpenCV and dlib for Python 2

```shell
# in current directory
git clone https://github.com/cmusatyalab/openface.git
cd openface
pip2 install -r requirements.txt
python2 setup.py install
models/get-mdoels.sh
cd ..
```

## download and preprocess dataset

Next download a dataset of face images.
It doesn’t matter if they have labels or not, we’ll get rid of them.
A non-exhaustive list of options are: MS-Celeb-1M, CelebA, CASIA-WebFace, FaceScrub, LFW, and MegaFace. 
Place the dataset in
`dcgan-completion.tensorflow/data/your-dataset/raw`
to indicate it’s the dataset’s raw images.

```shell
./openface/util/align-dlib.py data/dcgan-completion.tensorflow/data/your-dataset/raw align innerEyesAndBottomLip data/dcgan-completion.tensorflow/data/your-dataset/aligned --size 64
cd dcgan-completion.tensorflow/data/your-dataset/aligned
find . -name '*.png' -exec mv {} . \;
find . -type d -empty -delete
cd ../../..
```

## Start the training:

```shell
./train-dcgan.py --dataset ./data/your-dataset/aligned --epoch 20
```
