import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# matrix = height * width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


# defining the neural network

def neural_network_model(data):
    hiddenLayer1 = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hiddenLayer2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hiddenLayer3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    outputLayer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                   'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hiddenLayer1['weights']), hiddenLayer1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hiddenLayer2['weights']), hiddenLayer2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hiddenLayer3['weights']), hiddenLayer3['biases'])
    l3 = tf.nn.relu(l3)
    output = tf.add(tf.matmul(l3, outputLayer['weights']), outputLayer['biases'])
    return output


# training the network
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(prediction, tf.squeeze(y)))
    # cost = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    # cost = tf.reduce_mean(cost) * 100
    optimizer = tf.train.AdamOptimizer(0.003).minimize(cost)

    # cycles feed forward + backprop
    numberOfEpochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # dealing with training data
        for epoch in range(numberOfEpochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, ' completed out of ', numberOfEpochs, ' loss: ', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            print('Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)
