import tensorflow as tf

#  tf.version == 1.0.0
x = tf.Variable(35, name='x')
model_x = tf.variables_initializer([x])

y = tf.Variable(x + 5, name='y')
model_y = tf.variables_initializer([y])

with tf.Session() as session:
    session.run(model_x)
    session.run(model_y)
    print(session.run(y))
