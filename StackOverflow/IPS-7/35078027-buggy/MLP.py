import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
# one hidden layer MLP

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

W_h1 = tf.Variable(tf.random_normal([784, 512]))
h1 = tf.nn.sigmoid(tf.matmul(x, W_h1))

W_out = tf.Variable(tf.random_normal([512, 10]))
y_ = tf.matmul(h1, W_out)

# cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y_, y)
cross_entropy = tf.reduce_sum(- y * tf.log(y_) - (1 - y) * tf.log(1 - y_), 1)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train
with tf.Session() as s:
    s.run(tf.initialize_all_variables())

    for i in range(10000):
        batch_x, batch_y = mnist.train.next_batch(100)
        s.run(train_step, feed_dict={x: batch_x, y: batch_y})

        if i % 100 == 0:
            loss1, train_accuracy = s.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y})
            print('step {0}, training accuracy {1}, loss {2}'.format(i, train_accuracy, loss1))
