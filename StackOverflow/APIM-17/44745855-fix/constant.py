import tensorflow as tf

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)

x = tf.constant(35, name='x')
y = tf.Variable(x + 5, name='y')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))
