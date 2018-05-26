import tensorflow as tf

a = tf.Variable(1)
b = tf.Variable(1)
c = tf.Variable(2)
sum = tf.add(a, b)
with tf.control_dependencies([sum]):
    as0 = tf.assign(a, b)
    with tf.control_dependencies([as0]):
        as1 = tf.assign(b, c)
        with tf.control_dependencies([as1]):
            as2 = tf.assign(c, sum)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(10):
    print(sess.run([as2, as1, as0]))
