import tensorflow as tf

assert tf.__version__ >= "1.0"
tf.set_random_seed(20180130)

tf.merge_all_summaries()
print("Success!")