import tensorflow as tf
import numpy as np

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
np.random.seed(20180130)

def conv3d(inputs, weights, biases, layer_name, act=tf.nn.relu, padding='VALID'):
    preactivate = tf.nn.conv3d(inputs, weights, strides=[1, 1, 1, 1, 1], padding=padding) + biases
    activation = act(preactivate)
    return activation


def maxpool(inputs, padding='VALID'):
    return tf.nn.max_pool3d(inputs, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding=padding)


def weight_variable(shape, dtype=np.float32, partition_info=None):
    shape[shape == None] = 1
    n = np.prod(shape)
    w = (np.random.randn(n) * np.sqrt(2. / n)).astype(np.float32)
    return tf.Variable(w.reshape(shape), trainable=True)


## initializes biases
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, trainable=True)


def mean_square_error(a, b):
    shape = a.get_shape().as_list()
    shape[shape == None] = 1
    N = np.prod(shape)
    return tf.reduce_sum(tf.squared_difference(a, b)) / N


###  low level api ################################
def ll_model(inputs):
    input_layer = tf.reshape(inputs, [-1, 65, 65, 65, 2])

    W_conv1 = weight_variable([3, 3, 3, 2, 16])
    b_conv1 = bias_variable([16])
    conv1 = conv3d(input_layer, W_conv1, b_conv1, 'conv1')

    print(conv1.get_shape().as_list())

    pad = tf.pad(conv1, [[0, 0], [1, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
    print(pad.get_shape().as_list())

    maxpool1 = maxpool(pad)
    print(maxpool1.get_shape().as_list())

    W_conv2 = weight_variable([3, 3, 3, 16, 24])
    b_conv2 = bias_variable([24])
    conv2 = conv3d(maxpool1, W_conv2, b_conv2, 'conv2', padding="SAME")
    print(conv2.get_shape().as_list())

    W_conv3 = weight_variable([3, 3, 3, 24, 28])
    b_conv3 = bias_variable([28])
    conv3 = conv3d(conv2, W_conv3, b_conv3, 'conv3', padding="SAME")
    print(conv3.get_shape().as_list())

    W_conv4 = weight_variable([4, 4, 4, 28, 34])
    b_conv4 = bias_variable([34])
    conv4 = conv3d(conv3, W_conv4, b_conv4, 'conv4')
    print(conv4.get_shape().as_list())

    W_conv5 = weight_variable([5, 5, 5, 34, 42])
    b_conv5 = bias_variable([42])
    conv5 = conv3d(conv4, W_conv5, b_conv5, 'conv5')
    print(conv5.get_shape().as_list())

    W_conv6 = weight_variable([5, 5, 5, 42, 50])
    b_conv6 = bias_variable([50])
    conv6 = conv3d(conv5, W_conv6, b_conv6, 'conv6')
    print(conv6.get_shape().as_list())

    W_conv7 = weight_variable([5, 5, 5, 50, 50])
    b_conv7 = bias_variable([50])
    conv7 = conv3d(conv6, W_conv7, b_conv7, 'conv7')
    print(conv7.get_shape().as_list())

    W_conv8 = weight_variable([1, 1, 1, 50, 2])
    b_conv8 = bias_variable([2])
    conv8 = conv3d(conv7, W_conv8, b_conv8, 'output')

    return conv8


sess = tf.Session()
## placeholders
x = tf.placeholder(tf.float32, shape=[None, 65, 65, 65, 2], name='features')

y = tf.placeholder(tf.float32, shape=[None, 17, 17, 17, 2], name='targets')

loss = mean_square_error(y, ll_model(x))
