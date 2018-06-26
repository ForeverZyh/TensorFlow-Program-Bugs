import tensorflow as tf

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)

xs = [tf.constant(x) for x in range(10)]
dependency = None
dxs = []

for x in xs:
    if dependency is None:
        dependency = tf.Print(x, [x])
    else:
        with tf.control_dependencies([dependency]):
            dependency = tf.Print(x, [x])

    dxs.append(dependency)

print_all_op = tf.group(*dxs)

with tf.Session() as session:
    session.run(print_all_op)
