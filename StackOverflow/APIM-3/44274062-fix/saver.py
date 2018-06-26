import tensorflow as tf

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)

with tf.Graph().as_default():
    ####################################################
    # Architecture "A"
    w1 = tf.Variable(tf.linspace(0.0, 0.5, 6), name="w1")
    w2 = tf.Variable(tf.linspace(1.0, 5.0, 6), name="w2")

    saver = tf.train.Saver()  # <---------- Save everything

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver.save(sess, '/tmp/my_architecture')

with tf.Graph().as_default():
    ####################################################
    # Architecture "B"
    w1 = tf.Variable(tf.linspace(10.0, 50.0, 6), name="w1")
    w2 = tf.Variable(tf.linspace(100.0, 500.0, 6), name="w2")

    saver = tf.train.Saver({'w1': w1})
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, '/tmp/my_variable')

restored_graph = tf.Graph()
with restored_graph.as_default():
    tf.train.import_meta_graph('/tmp/my_architecture.meta')
    vars_to_restore = [
        restored_graph.get_tensor_by_name(var_name + ':0') for var_name, _
        in tf.contrib.framework.list_variables('/tmp/my_variable')]
    filtered_saver = tf.train.Saver(var_list=vars_to_restore)
    with tf.Session() as sess:
        # Restore w1 from Architecture "B" into the metagraph from Architecture "A"
        filtered_saver.restore(sess, '/tmp/my_variable')
        print(restored_graph.get_tensor_by_name('w1:0').eval())