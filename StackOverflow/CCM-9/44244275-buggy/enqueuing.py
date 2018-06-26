from __future__ import division, print_function, unicode_literals
import math
import numpy
import tensorflow as tf
from tensorflow.python.training import queue_runner
from tensorflow.python.ops import control_flow_ops

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
numpy.random.seed(20180130)

row_count, column_count = 7, 5
batch_size, step_size = 3, 2

# Create some random data
data = numpy.arange(row_count * column_count).reshape(
    (row_count, column_count))
print(data)

batch_count = int(math.ceil(row_count / batch_size))
step_count = int(math.ceil(column_count / step_size))
print(batch_count, step_count)

slices = tf.train.slice_input_producer([data], num_epochs=1, shuffle=False)
batch = tf.train.batch(slices, batch_size, allow_smaller_final_batch=True)

queue = tf.FIFOQueue(32, dtypes=[batch.dtype])
enqueue_ops = []
dependency = None

for step_index in range(step_count):
    step = tf.strided_slice(
        batch, [0, step_index * step_size],
        [tf.shape(batch)[0], (step_index + 1) * step_size])

    if dependency is None:
        dependency = step
    else:
        step = control_flow_ops.with_dependencies([dependency], step)

    enqueue_ops.append(queue.enqueue(step))

queue_runner.add_queue_runner(queue_runner.QueueRunner(
    queue=queue, enqueue_ops=[tf.group(*enqueue_ops)]))
step = queue.dequeue()

supervisor = tf.train.Supervisor()

with supervisor.managed_session() as session:
    for batch_index in range(batch_count):
        for step_index in range(step_count):
            print("Batch %d, step %d" % (batch_index, step_index))
            print(session.run(step))
