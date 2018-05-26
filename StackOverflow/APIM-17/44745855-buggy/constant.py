import tensorflow as tf

x = tf.constant(35, name='x')
y = tf.constant(x + 5, name='y')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))
