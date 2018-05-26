import tensorflow as tf

with tf.Session() as session:
    for i in range(5):
        x = tf.Variable(0, name='x')
        model = tf.global_variables_initializer()
        session.run(model)
        x = x + 1
        print(session.run(x))
