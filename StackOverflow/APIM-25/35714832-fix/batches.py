import tensorflow as tf

sess = tf.Session()


def add_to_batch(image):
    print('Adding to batch')
    image_batch = tf.train.shuffle_batch([image], batch_size=5, capacity=11, min_after_dequeue=1, num_threads=1)

    # Add to summary
    tf.summary.image('images', image_batch)

    return image_batch


def get_batch():
    # Create filename queue of images to read
    filenames = ["sample.jpg" for _ in range(11)]
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    # Read and process image
    my_image = tf.image.decode_png(value)
    my_image_float = tf.cast(my_image, tf.float32)
    image_mean = tf.reduce_mean(my_image_float)
    my_noise = tf.random_normal([959, 959, 3], mean=image_mean)
    my_image_noisy = my_image_float + my_noise
    print('Reading images')

    return add_to_batch(my_image_noisy)


def main():
    sess.run(tf.initialize_all_variables())
    writer = tf.summary.FileWriter('./summary_logs', graph_def=sess.graph_def)
    images = get_batch()
    tf.train.start_queue_runners(sess=sess)
    merged = tf.summary.merge_all()
    summary_str = sess.run(merged)
    writer.add_summary(summary_str)
    print("Start training")


main()
