import tensorflow as tf

# Model parameters
A = tf.Variable([0], dtype=tf.float32)
B = tf.Variable([0], dtype=tf.float32)
C = tf.Variable([0], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
model = A * (x ** 2) + B * x + C
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(model - y))  # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

# training data
x_train = [0, 1, 2, 3]
y_train = [0, 1, 4, 9]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(100):
    sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_A, curr_B, curr_C, curr_loss = sess.run([A, B, C, loss], {x: x_train, y: y_train})
print("A: %s B: %s C: %s loss: %s" % (curr_A, curr_B, curr_C, curr_loss))
