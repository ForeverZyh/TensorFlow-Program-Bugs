import tensorflow as tf

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)

table = tf.contrib.lookup.MutableHashTable(key_dtype=tf.string,
                                           value_dtype=tf.float32,
                                           default_value=-1)
key = tf.constant('hi', tf.string)
val = tf.constant(1.1, tf.float32)
table.insert(key, val)

sess = tf.InteractiveSession()
print(table.lookup(key).eval())
