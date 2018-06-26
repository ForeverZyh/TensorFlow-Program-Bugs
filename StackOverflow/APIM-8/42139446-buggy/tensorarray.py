import tensorflow as tf

assert tf.__version__ == "1.0.0"
tf.set_random_seed(20180130)

aI = tf.TensorArray(tf.int32, 2)
aO = tf.TensorArray(tf.int32, 2)
aI = aI.unstack([[1, 2], [1, 2]])


def body(i, aI, aO):
    aO.write(i, aI.read(i) + 1)
    return (i + 1, aI, aO)


cond = lambda i, *_: i < 2
_, _, aO = tf.while_loop(cond, body, [0, aI, aO])
r = aO.stack()
with tf.Session() as sess:
    res = sess.run(r)
    print('done!')
