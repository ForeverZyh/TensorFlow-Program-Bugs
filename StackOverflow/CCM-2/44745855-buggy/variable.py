import tensorflow as tf

#  tf.version == 1.0.0
x = tf.Variable(35, name='x')
y = tf.Variable(x + 5, name='y')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))
