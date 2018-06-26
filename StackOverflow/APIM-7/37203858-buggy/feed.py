import tensorflow as tf
import numpy as np

assert tf.__version__ == "1.8.0"
np.random.seed(20180130)
tf.set_random_seed(20180130)

n_data = 1000
features = np.random.uniform(0, 1, [n_data, 10])
labels = np.random.uniform(0, 1, [n_data, 4])


def read_data():
    BATCH_SIZE = 16
    capacity = BATCH_SIZE * 10
    min_after_dequeue = BATCH_SIZE * 5
    data_batch, label_batch = tf.train.shuffle_batch([features, labels], batch_size=BATCH_SIZE,
                                                     capacity=capacity, min_after_dequeue=min_after_dequeue,
                                                     enqueue_many=True)
    return data_batch, label_batch


with tf.Session() as sess:
    train_data_node = tf.placeholder(tf.float32, [None, 10])
    train_labels_node = tf.placeholder(tf.float32, [None, 4])
    W = tf.Variable(tf.truncated_normal([10, 4]))
    B = tf.Variable(tf.truncated_normal([4]))
    logits = tf.matmul(train_data_node, W) + B
    loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_labels_node))
    optimizer = tf.train.AdamOptimizer().minimize(loss_value)

    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print('Initialized!')
    for step in range(1000):
        data, label = read_data()
        feed_dict = {train_data_node: data,
                     train_labels_node: label}
        _, l = sess.run([optimizer, loss_value], feed_dict=feed_dict)
        print("loss: ", l)
