import tensorflow as tf
from tqdm import tqdm

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)

TRAIN_DATA_PATH = './heart_train.csv'
TEST_DATA_PATH = './heart_test.csv'
BATCH_SIZE = 7
N_FEATURES = 9
BETA = 1  # regularizer
EPOCHS = 1000
LEARNING_RATE = 0.5

import pandas as pd

pd_train_data = pd.read_csv(TRAIN_DATA_PATH)
n_train_data, _ = pd_train_data.shape
print(n_train_data)
pd_test_data = pd.read_csv(TEST_DATA_PATH)
n_test_data, _ = pd_test_data.shape
print(n_test_data)


# Step 1: get data

def batch_generator(filenames):
    """ filenames is the list of files you want to read from.
    In this case, it contains only heart.csv
    """
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TextLineReader(skip_header_lines=1)  # skip the first line in the file
    _, value = reader.read(filename_queue)

    # record_defaults are the default values in case some of our columns are empty
    # This is also to tell tensorflow the format of our data (the type of the decode result)
    # for this dataset, out of 9 feature columns,
    # 8 of them are floats (some are integers, but to make our features homogenous,
    # we consider them floats), and 1 is string (at position 5)
    # the last column corresponds to the lable is an integer

    record_defaults = [[1.0] for _ in range(N_FEATURES)]
    record_defaults[4] = ['']
    record_defaults.append([1])

    # read in the 10 rows of data
    content = tf.decode_csv(value, record_defaults=record_defaults)

    # convert the 5th column (present/absent) to the binary value 0 and 1
    condition = tf.equal(content[4], tf.constant('Present'))
    content[4] = tf.cond(condition, lambda: tf.constant(1.0), lambda: tf.constant(0.0))

    # pack all 9 features into a tensor
    features = tf.stack(content[:N_FEATURES])
    # assign the last column to label
    labels = tf.stack([1 - content[-1], content[-1]], 0)

    # minimum number elements in the queue after a dequeue, used to ensure
    # that the samples are sufficiently mixed
    # I think 10 times the BATCH_SIZE is sufficient
    min_after_dequeue = 10 * BATCH_SIZE

    # the maximum number of elements in the queue
    capacity = 20 * BATCH_SIZE

    # shuffle the data to generate BATCH_SIZE sample pairs
    data_batch, label_batch = tf.train.shuffle_batch([features, labels], batch_size=BATCH_SIZE,
                                                     capacity=capacity, min_after_dequeue=min_after_dequeue)

    return data_batch, label_batch


def generate_batches(data_batch, label_batch):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for _ in range(10):  # generate 10 batches
            features, labels = sess.run([data_batch, label_batch])
            print(features, labels)
        coord.request_stop()
        coord.join(threads)


data1_feature_batch, data1_label_batch = batch_generator([TRAIN_DATA_PATH])
test_data1_feature_batch, test_data1_label_batch = batch_generator([TEST_DATA_PATH])

# generate_batches(data1_feature_batch, data1_label_batch)
# generate_batches(test_data1_feature_batch, test_data1_label_batch)

# Step 2: create placeholders for input X (Features) and label Y (binary result)
X = tf.placeholder(tf.float32, shape=[None, 9], name="X")
Y = tf.placeholder(tf.float32, shape=[None, 2], name="Y")

# Step 3: create weight and bias, initialized to 0
w = tf.Variable(tf.truncated_normal([9, 2]), name="weights")
b = tf.Variable(tf.zeros([1, 2]), name="bias")

# Step 4: logistic multinomial regression / softmax
score = tf.matmul(X, w) + b

# Step 5: define loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=Y, name="entropy")

regularizer = tf.nn.l2_loss(w)
loss = tf.reduce_mean(entropy + BETA * regularizer, name="loss")

# Step 6: using gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

# Step 7: Prediction
Y_predicted = tf.nn.softmax(tf.matmul(X, w) + b)
correct_prediction = tf.equal(tf.argmax(Y_predicted, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train():
    import glob, os

    for f in glob.glob("/tmp/model.ckpt*"):
        os.remove(f)

    saver = tf.train.Saver([w, b])
    EPOCHS = 1000
    with tf.device("/cpu:0"):
        with tf.Session() as sess:
            # Step 7: initialize the necessary variables, in this case, w and b
            sess.run(tf.global_variables_initializer())

            # Step 8: train the model
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            n_batches = int(n_train_data / BATCH_SIZE)
            for epoch in tqdm(range(EPOCHS)):  # run epochs
                avg_loss = avg_acc = 0

                for _ in range(n_batches):
                    x_batch, y_batch = sess.run([data1_feature_batch, data1_label_batch])
                    # Session runs train_op to minimize loss
                    feed_dict = {X: x_batch, Y: y_batch}
                    _, loss_batch, acc = sess.run([optimizer, loss, accuracy], feed_dict=feed_dict)
                    avg_loss += loss_batch / n_batches
                    avg_acc += acc / n_batches

                if (epoch + 1) % 100 == 0:
                    print("avg_loss", avg_loss)
                    print("avg_acc", avg_acc)

            coord.request_stop()
            coord.join(threads)

            # Step 9: saving the values of w and b
            print("weights", w.eval())
            print("bias", b.eval())

            # Add ops to save and restore all the variables.
            save_path = saver.save(sess, "/tmp/logit_reg_tf_model")


train()
