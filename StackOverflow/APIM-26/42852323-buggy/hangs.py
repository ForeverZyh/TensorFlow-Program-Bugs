import tensorflow as tf

x1 = [1, 3, 4, 5, 1, 6, -1, -3]
x2 = [5, 2, 1, 5, 0, 2, 4, 2]
y = [6, 5, 5, 10, 1, 8, 3, -1]


def train_fn():
    return {'x1': tf.constant(x1), 'x2': tf.constant(x2)}, tf.constant(y)


features = [tf.contrib.layers.real_valued_column('x1', dimension=1),
            tf.contrib.layers.real_valued_column('x2', dimension=1)]
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
estimator.fit(input_fn=train_fn, steps=10000)

for vn in estimator.get_variable_names():
    print('variable name', vn, estimator.get_variable_value(vn))
print(estimator.evaluate(input_fn=train_fn))
