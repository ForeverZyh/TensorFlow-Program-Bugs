import tensorflow as tf

n_input = 10
n_classes = 4
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
W = tf.Variable(tf.truncated_normal([n_input, n_classes]))
B = tf.Variable(tf.truncated_normal([n_classes]))
pred = tf.matmul(x, W) + B
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
encoder = {"apple": 0, "orange": 1, "banana": 2}
input_y = [["apple", "apple", "orange", "banana"]]
input_y_ = [[encoder[yy] for yy in xx] for xx in input_y]
input_x = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
sess.run(cost, feed_dict={x: input_x, y: input_y_})
