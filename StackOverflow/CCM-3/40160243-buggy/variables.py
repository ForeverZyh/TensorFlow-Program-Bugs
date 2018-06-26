import tensorflow as tf
import numpy as np

assert tf.__version__ == "1.0.0"
tf.set_random_seed(20180130)

data = np.random.randint(1000, size=10000)
x = tf.Variable(data, name='x')
y = tf.Variable(5 * x * x - 3 * x + 15, name='y')

model = tf.initialize_all_variables()

with tf.Session() as s:
    s.run(model)
    print(s.run(y))
