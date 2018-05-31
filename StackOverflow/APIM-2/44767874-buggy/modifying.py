import tensorflow as tf
import time

tf.set_random_seed(seed=0)
global_step = tf.constant(100)
training_steps = tf.Variable(0, trainable=False)
supervisor = tf.train.Supervisor(logdir="./log",
                                 global_step=global_step)

# create coordinator to handle threading
coord = tf.train.Coordinator()

with supervisor.managed_session() as sess:
    # start threads to enqueue input minibatches for training
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # initialize all variables and ops
    sess.run(tf.global_variables_initializer())

    start_time = time.time()

    # train
    for step in range(1, training_steps):

        # check for any raised exceptions
        if supervisor.should_stop():
            break

    # when done, ask the threads to stop
    coord.request_stop()

    # wait for threads to finish
    coord.join(threads)
