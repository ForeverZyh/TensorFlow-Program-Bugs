import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)

xs = [tf.constant(x) for x in range(10)]
xs = [tf.Print(x, [x]) for x in xs]
dependency = None
dxs = []

for x in xs:
    if dependency is None:
        dependency = x
    else:
        dependency = control_flow_ops.with_dependencies([dependency], x)

    dxs.append(dependency)

print_all_op = tf.group(*dxs)

with tf.Session() as session:
    session.run(print_all_op)
