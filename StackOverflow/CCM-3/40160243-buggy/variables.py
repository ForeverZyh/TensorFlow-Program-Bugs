import tensorflow as tf
import numpy as np

#  tf.version == 1.0.0
data = np.random.randint(1000, size=10000)
x = tf.Variable(data, name='x')
y = tf.Variable(5 * x * x - 3 * x + 15, name='y')

model = tf.initialize_all_variables()

with tf.Session() as s:
    s.run(model)
    print(s.run(y))
