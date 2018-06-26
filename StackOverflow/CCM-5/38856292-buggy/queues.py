import tensorflow as tf

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)

fifo_q = tf.FIFOQueue(
    capacity=10,
    dtypes=[tf.string, tf.string],
    shapes=[[], []])

sess = tf.Session()

l = [str(i + 1) for i in range(10)]
x = tf.constant(l)
y = tf.constant(l)

eq = fifo_q.enqueue_many([x, y])
dq1, dq2 = fifo_q.dequeue()
sess.run(eq)

for x in range(5):
    print('dq1:', sess.run(dq1))
    print('dq2:', sess.run(dq2))
