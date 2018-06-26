import tensorflow as tf

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)

pred = tf.placeholder(tf.bool)
x = tf.Variable([1])
assign_x_2 = tf.assign(x, [2])


def update_x_2():
    with tf.control_dependencies([assign_x_2]):
        return tf.identity(x)


y = tf.cond(pred, update_x_2, lambda: tf.identity(x))
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    print(y.eval(feed_dict={pred: False}))
    print(y.eval(feed_dict={pred: True}))
