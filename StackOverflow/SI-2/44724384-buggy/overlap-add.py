import tensorflow as tf

assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)

input = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]


def overlap_add(overlap):
    with tf.Graph().as_default(), tf.Session() as sess:
        x = tf.constant(input)

        num_chunks = tf.shape(x)[0]
        chunk_size = tf.shape(x)[1]
        hop_length = chunk_size - overlap
        out_len = chunk_size + hop_length * (num_chunks - 1)

        y = tf.zeros((out_len,), dtype=tf.int32)

        def body(i, y):
            j = i * hop_length
            padding = [[j, out_len - (j + chunk_size)]]
            chunk = x[i]
            y = y + tf.pad(chunk, padding)
            return (i + 1, y)

        i = tf.constant(0)
        i, y = tf.while_loop(
            cond=lambda i, _: tf.less(i, num_chunks),
            body=body,
            loop_vars=[i, y])

        return sess.run(y)


for i in range(4):
    print('overlap_add(%d): %s' % (i, overlap_add(i)))

# overlap_add(0): [ 1  2  3  4  5  6  7  8  9 10 11 12]
# overlap_add(1): [ 1  2  3  9  6  7 17 10 11 12]
# overlap_add(2): [ 1  2  8 10 16 18 11 12]
# overlap_add(3): [ 1  7 18 21 19 12]
