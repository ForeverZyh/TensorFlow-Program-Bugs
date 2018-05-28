import tensorflow as tf

input = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]


def tf_repeat(a, repeats):
    return tf.reshape(tf.tile(tf.reshape(a, [-1, 1]),
                              [1, repeats]), [-1])


def overlap_add(overlap):
    with tf.Graph().as_default(), tf.Session() as sess:
        x = tf.constant(input)
        x_flat = tf.reshape(x, [-1])

        num_chunks = tf.shape(x)[0]
        chunk_size = tf.shape(x)[1]
        hop_len = chunk_size - overlap
        flat_len = num_chunks * chunk_size
        out_len = chunk_size + hop_len * (num_chunks - 1)

        # e.g. [0,1,2,3, 2,3,4,5, 4,5,6,7] for overlap == 2
        indexes = tf.range(flat_len) - tf_repeat(tf.range(num_chunks), chunk_size) * overlap

        return sess.run(tf.unsorted_segment_sum(x_flat, indexes, out_len))


for i in range(4):
    print('overlap_add(%d): %s' % (i, overlap_add(i)))

# overlap_add(0): [ 1  2  3  4  5  6  7  8  9 10 11 12]
# overlap_add(1): [ 1  2  3  9  6  7 17 10 11 12]
# overlap_add(2): [ 1  2  8 10 16 18 11 12]
# overlap_add(3): [ 1  7 18 21 19 12]
