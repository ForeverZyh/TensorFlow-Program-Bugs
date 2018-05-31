import tensorflow as tf

####################################################
# Architecture "A"
w1 = tf.Variable(tf.linspace(0.0, 0.5, 6), name="w1")
w2 = tf.Variable(tf.linspace(1.0, 5.0, 6), name="w2")

saver = tf.train.Saver()  # <---------- Save everything

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver.save(sess, '/tmp/my_architecture')

tf.reset_default_graph()
####################################################
# Architecture "B"
w1 = tf.Variable(tf.linspace(10.0, 50.0, 6), name="w1")
w2 = tf.Variable(tf.linspace(100.0, 500.0, 6), name="w2")

saver = tf.train.Saver({'w1': w1})
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver.save(sess, '/tmp/my_variable')

tf.reset_default_graph()
######################################################
with tf.Session() as sess:
    # Loading the model structure from 'my_test_model.meta'
    new_saver = tf.train.import_meta_graph('/tmp/my_architecture.meta')

    # Loading the saved "w1" Variable
    new_saver.restore(sess, '/tmp/my_variable')
