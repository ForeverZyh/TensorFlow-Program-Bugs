import pickle
import re
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import *

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
np.random.seed(20180130)
random.seed(20180130)

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def vgg16Net(inputs,
             num_classes=1000,
             is_training=True,
             dropout_keep_prob=0.5,
             spatial_squeeze=True,
             scope='vgg_16'):
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d
        with framework.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                 outputs_collections=end_points_collection):
            net = layers.repeat(inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
            net = layers.max_pool2d(net, [2, 2], scope='pool1')
            net = layers.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
            net = layers.max_pool2d(net, [2, 2], scope='pool2')
            net = layers.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
            net = layers.max_pool2d(net, [2, 2], scope='pool3')
            net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
            net = layers.max_pool2d(net, [2, 2], scope='pool4')
            net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
            net = layers.max_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            net = layers.conv2d(net, 4096, [7, 10], padding='VALID', scope='fc6')
            net = layers.dropout(net, dropout_keep_prob, is_training=is_training,
                                 scope='dropout6')
            net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
            net = layers.dropout(net, dropout_keep_prob, is_training=is_training,
                                 scope='dropout7')
            net = layers.conv2d(net, num_classes, [1, 1],
                                activation_fn=None,
                                normalizer_fn=None,
                                scope='fc8')
            # Convert end_points_collection into a end_point dict.
            end_points = tf.contrib.slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
            return net, end_points


def getsamples():
    n = 100
    return np.asarray(["sample.jpg" for _ in range(n)]), np.asarray([random.randint(0, 99) for _ in range(n)])


samples_all, labels_all = getsamples()

numOfSamples = len(samples_all)
labels_one_hot_all = np.zeros((len(labels_all), 101))
index_offset = np.arange(len(labels_all)) * 101
ind = index_offset + np.array(labels_all, np.int32) - 1
labels_one_hot_all.flat[ind] = 1
# samples_all = tf.constant(samples_all)
# labels_all = tf.constant(labels_all)
# samples_all_p = tf.placeholder(dtype = tf.string, shape = (numOfSamples,) )
# labels_one_hot_all_p = tf.placeholder(dtype = tf.float32, shape = (numOfSamples, 101))
# samples_all_v = tf.Variable(np.asarray(['']*numOfSamples), name = 'sample', trainable = False)
# labels_one_hot_all_v = tf.Variable(np.zeros_like(labels_one_hot_all, dtype=np.float32), name = 'label', trainable = False)
[sample, label] = tf.train.slice_input_producer([samples_all, labels_one_hot_all])
imagecontent = tf.read_file(sample)
image = tf.image.decode_jpeg(imagecontent, channels=3)
image = tf.cast(image, dtype=tf.float32)
# channels = tf.split(2, 3, image)
# channels[0] -= _R_MEAN
# channels[1] -= _G_MEAN
# channels[2] -= _B_MEAN
# image = tf.concat(2, channels)
image = tf.reshape(tf.image.resize_images(image, [240, 320]), [240, 320, 3])
images, labels = tf.train.batch([image, label], 16, 3, 32)
net, end = vgg16Net(images, num_classes=101, is_training=True)

losses.softmax_cross_entropy(net, labels)
total_loss = losses.get_total_loss()

global_step = tf.Variable(0, trainable=False, name='global_step')
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)

train_var = framework.get_variables_to_restore(
    exclude=['vgg_16/conv1', 'vgg_16/conv2', 'vgg_16/conv2', 'vgg_16/conv3', 'vgg_16/conv4', 'vgg_16/conv5',
             'global_step', 'sample', 'label'])
init_var = framework.get_variables_to_restore(
    exclude=['vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8', 'global_step', 'sample', 'label'])
# init_op, feed_init = framework.assign_from_checkpoint('./vgg_16.ckpt', init_var)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(total_loss, global_step, var_list=train_var)

# with tf.name_scope('accuracy'):
#    with tf.name_scope('correct_prediction'):
#      correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(labels, 1))
#    with tf.name_scope('accuracy'):
#      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# summaries
# model_store_dir = 'C:\\Users\\mx\\Desktop\\nextLevel\\nextLevel\\nextLevel\\log1\\'
summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
# summaries.add(tf.summary.scalar('accuracy', accuracy))
for end_point in end:
    x = end[end_point]
    summaries.add(tf.summary.histogram('activations/' + end_point, x))
    summaries.add(tf.summary.scalar('sparsity/' + end_point, tf.nn.zero_fraction(x)))

for loss in tf.get_collection(tf.GraphKeys.LOSSES):
    summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

for variable in framework.get_model_variables():
    summaries.add(tf.summary.histogram(variable.op.name, variable))

summaries.add(tf.summary.scalar('learning_rate', learning_rate))

summaries.add(tf.summary.scalar('total_loss', total_loss))

# summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))

summary_op = tf.summary.merge(list(summaries))

# summary_writer = tf.summary.FileWriter(model_store_dir)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # sess.run(init_op, feed_dict=feed_init)
    for i in range(100000):
        if i % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            s, _ = sess.run([summary_op, training_op],
                            options=run_options,
                            run_metadata=run_metadata)
            # summary_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            # summary_writer.add_summary(s, i)
            print('Adding run metadata for', i)
        else:
            s, _ = sess.run([summary_op, training_op])
            print("Training")
