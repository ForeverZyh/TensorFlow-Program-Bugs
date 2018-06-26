import tensorflow as tf

assert tf.__version__[:len("0.12.")] == "0.12."
tf.set_random_seed(20180130)

label_bytes = 1
record_bytes = tf.random_normal(dtype=tf.float32, shape=[100, 10], mean=0, stddev=1)
label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes], [1]), tf.int32)
sess = tf.Session()
print(sess.run(label))
