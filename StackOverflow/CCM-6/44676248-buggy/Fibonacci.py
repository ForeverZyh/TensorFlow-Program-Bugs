import tensorflow as tf

a = tf.Variable(1)
b = tf.Variable(1)
c = tf.Variable(2)
sum = tf.add(a, b)

as0 = tf.assign(a, b)
as1 = tf.assign(b, c)
as2 = tf.assign(c, sum)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(10):
    print(sess.run([as2, as1, as0]))
