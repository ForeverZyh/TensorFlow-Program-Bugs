import tensorflow as tf

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)

with tf.Session() as session:
    for i in range(5):
        x = tf.Variable(0, name='x')
        model = tf.global_variables_initializer()
        session.run(model)
        x = x + 1
        print(session.run(x))
