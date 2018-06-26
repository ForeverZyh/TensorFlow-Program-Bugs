import tensorflow as tf
import numpy as np

assert tf.__version__[:len("0.12.")] == "0.12."
np.random.seed(20180130)
tf.set_random_seed(20180130)

features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, batch_size=4, num_epochs=1000)

estimator.fit(input_fn=input_fn, steps=1000)

print(estimator.evaluate(input_fn=input_fn))
