import tensorflow as tf

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)

def weight_variable(shape):
    """Initialize the weights with random weights"""
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(initial)


# Initialize my data
x = tf.constant([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=tf.float64)
y_ = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)

w = weight_variable((2, 1))
y = tf.matmul(x, w)

error = tf.reduce_mean(tf.square(y_ - y))

train_step = tf.train.AdamOptimizer(1e-4).minimize(error)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    # Train the model and output every 1000 iterations
    for i in range(50000):
        sess.run(train_step)
        err = sess.run(error)

        if i % 1000 == 0:
            print("\nerr:", err)
            print("x: ", sess.run(x))
            print("w: ", sess.run(w))
            print("y_: ", sess.run(y_))
            print("y: ", sess.run(y))
