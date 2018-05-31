import tensorflow as tf
import numpy as np
import time

batch_size = 32
max_steps = 1000


def inference(x):
    x_ = tf.layers.flatten(x)
    W = tf.Variable(tf.truncated_normal([33 * 33, 21 * 21]))
    tf.summary.histogram("W", W)
    B = tf.Variable(tf.truncated_normal([21 * 21]))
    tf.summary.histogram("B", B)
    logits = tf.matmul(x_, W) + B
    tf.summary.histogram("logits", logits)
    return logits


def train(losses, global_step):
    return tf.train.AdamOptimizer().minimize(loss=losses, global_step=global_step)


def loss(logits, _labels):
    labels = tf.layers.flatten(_labels)
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="losses")
    tf.summary.histogram("losses", losses)
    return losses


with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    images = tf.placeholder(tf.float32, shape=[batch_size, 33, 33, 1])
    labels = tf.placeholder(tf.float32, shape=[batch_size, 21, 21, 1])

    logits = inference(images)
    losses = loss(logits, labels)
    train_op = train(losses, global_step)
    saver = tf.train.Saver(tf.all_variables())
    summary_op = tf.summary.merge_all()
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    summary_writer = tf.summary.FileWriter("./train", sess.graph)

    for step in range(max_steps):
        start_time = time.time()

        data_batch, label_batch = np.random.normal(0, 1, [batch_size, 33, 33, 1]), np.random.uniform(0, 1,
                                                                                                     [batch_size, 21,
                                                                                                      21,
                                                                                                      1])
        _, loss_value = sess.run([train_op, losses],
                                 feed_dict={images: data_batch, labels: label_batch})

        summary_str = sess.run(summary_op)

        duration = time.time() - start_time
