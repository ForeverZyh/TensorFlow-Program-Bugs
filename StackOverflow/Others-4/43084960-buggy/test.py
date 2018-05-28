import argparse
import logging

import tensorflow as tf

log = logging.getLogger(__name__)

# Job Names
PARAMETER_SERVER = "ps"
WORKER_SERVER = "worker"

# Cluster Details
CLUSTER_SPEC = {
    PARAMETER_SERVER: ["localhost:2222"],
    WORKER_SERVER: ["localhost:1111", "localhost:1112"]}


def parse_command_arguments():
    """ Set up and parse the command line arguments passed for experiment. """
    parser = argparse.ArgumentParser(
        description="Parameters and Arguments for the Test.")
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )

    return parser.parse_args()


def start_server(job_name, task_index):
    """ Create a server based on a cluster spec. """
    cluster = tf.train.ClusterSpec(CLUSTER_SPEC)
    server = tf.train.Server(
        cluster, job_name=job_name, task_index=task_index)

    return server, cluster


def model():
    """ Build up a simple estimator model. """
    # Build a linear model and predict values
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    # Loss sub-graph
    loss = tf.reduce_sum(tf.square(linear_model - y))

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss, global_step=global_step)

    init_op = tf.global_variables_initializer()
    log.info("Variables initialized ...")

    return W, b, loss, x, y, train, global_step, init_op


if __name__ == "__main__":
    # Initializing logging with level "INFO".
    logging.basicConfig(level=logging.INFO)

    # Parse arguments from command line.
    arguments = parse_command_arguments()
    job_name = arguments.job_name
    task_index = arguments.task_index

    # Start a server.
    server, cluster = start_server(job_name, task_index)

    if job_name == "ps":
        server.join()
    else:
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):
            W, b, loss, x, y, train, global_step, init_op = model()
        with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=(arguments.task_index == 0 and (
                        arguments.job_name == 'worker'))) as sess:
            step = 0
            # training data
            x_train = [1, 2, 3, 4]
            y_train = [0, -1, -2, -3]
            while not sess.should_stop() and step < 1000:
                _, step = sess.run(
                    [train, global_step], {x: x_train, y: y_train})

            # evaluate training accuracy
            curr_W, curr_b, curr_loss = sess.run(
                [W, b, loss], {x: x_train, y: y_train})
            print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
