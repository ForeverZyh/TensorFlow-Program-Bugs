import tensorflow as tf

# tf.version == 0.12.1rc
label_bytes = 1
record_bytes = tf.constant(0, dtype=tf.int8, shape=[100, 10])
label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
