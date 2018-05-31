import tensorflow as tf

x = tf.placeholder(tf.float32, [None])
y = tf.placeholder(tf.float32, [None])
W = tf.Variable(tf.zeros([1]))
B = tf.Variable(tf.zeros([1]))
y_ = tf.multiply(x, W) + B
loss = tf.reduce_mean(tf.square(y_ - y))

# Add the ops to initialize variables.  These will include
# the optimizer slots added by AdamOptimizer().
init_op = tf.initialize_all_variables()

# Add the optimizer
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

# launch the graph in a session
sess = tf.Session()
# Actually intialize the variables
sess.run(init_op)
# now train your model
for i in range(100):
    sess.run(train_op, feed_dict={x: [1], y: [2]})
