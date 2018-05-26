import tensorflow as tf

pred = tf.placeholder(tf.bool)
x = tf.Variable([1])


def update_x_2():
    with tf.control_dependencies([tf.assign(x, [2])]):
        return tf.identity(x)


y = tf.cond(pred, update_x_2, lambda: tf.identity(x))
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    print(y.eval(feed_dict={pred: False}))
    print(y.eval(feed_dict={pred: True}))
