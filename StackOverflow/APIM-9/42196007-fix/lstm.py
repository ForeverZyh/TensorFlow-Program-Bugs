import tensorflow as tf

length = 30
lstm = tf.nn.rnn_cell.BasicLSTMCell(128)
cell = tf.nn.rnn_cell.MultiRNNCell([lstm] * 2)
state = cell.zero_state(1, tf.float32)

inputImgA = tf.TensorArray(tf.string, length)
outputLSTM = tf.TensorArray(tf.float32, length)
lossLSTM = tf.TensorArray(tf.float32, length)

i = tf.constant(0)


def cond(i, state, inputImgA, outputLSTM, lossLSTM):
    return tf.less(i, length)


n_classes = 4
label = tf.constant([0])


def body(i, state, inputImgA, outputLSTM, lossLSTM):
    imcontent = inputImgA.read(i)
    image = tf.constant(value=0, shape=[224, 224, 3], dtype=tf.float32)
    with tf.variable_scope('Image_Process'):
        image = tf.image.resize_images(image, [224, 224])
        images = tf.expand_dims(image, 0)

    W = tf.Variable(lambda: tf.truncated_normal([224 * 224 * 3, n_classes]))
    B = tf.Variable(lambda: tf.truncated_normal([n_classes]))
    output = tf.matmul(tf.reshape(images, [1, 224 * 224 * 3]), W) + B

    outputLSTM = outputLSTM.write(i, output)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=label)
    lossLSTM = lossLSTM.write(i, loss + tf.nn.l2_loss(W))
    return (i + 1, state, inputImgA, outputLSTM, lossLSTM)


_, _, _, outputLSTM, lossLSTM = tf.while_loop(cond, body, [i, state, inputImgA, outputLSTM, lossLSTM])

output = outputLSTM.stack()
loss = lossLSTM.stack()
loss = tf.reduce_mean(loss)
losses = tf.losses
# losses.add_loss(tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
losses.add_loss(loss)
total_loss = losses.get_total_loss()
