import tensorflow as tf

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)

filenames = ['./sample.jpg']
labels = [1]

filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()
filename, content = reader.read(filename_queue)
images = tf.image.decode_jpeg(content, channels=3)
images = tf.cast(images, tf.float32)
resized_images = tf.image.resize_images(images, [224, 224])

image_batch, label_batch = tf.train.batch([resized_images, labels], batch_size=2)
print(image_batch)