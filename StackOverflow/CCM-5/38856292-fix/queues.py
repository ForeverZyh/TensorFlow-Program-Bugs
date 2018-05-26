import tensorflow as tf

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
    dq1_val, dq2_val = sess.run([dq1, dq2])
    print('dq1:', dq1_val)
    print('dq2:', dq2_val)
