import tensorflow as tf

assert tf.__version__ >= "1.0"
tf.set_random_seed(20180130)

tf.summary.merge_all()
print("Success!")