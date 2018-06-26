import tensorflow as tf

# assert tf.__version__ == "0.5.0"
# 0.5.0 version tensorflow even does not have the attribute '__version__'
tf.set_random_seed(20180130)

print(tf.compat.as_str("tensorflow bugs").split())
