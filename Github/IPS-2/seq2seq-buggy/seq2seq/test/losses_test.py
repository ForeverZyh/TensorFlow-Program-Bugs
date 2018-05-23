"""
Unit tests for loss-related operations.
"""

from seq2seq import losses as seq2seq_losses
import tensorflow as tf
import numpy as np


class CrossEntropySequenceLossTest(tf.test.TestCase):
  """
  Test for `sqe2seq.losses.sequence_mask`.
  """

  def setUp(self):
    super(CrossEntropySequenceLossTest, self).setUp()
    tf.logging.set_verbosity(tf.logging.INFO)
    self.batch_size = 4
    self.sequence_length = 10
    self.vocab_size = 50

  def test_op(self):
    logits = np.random.randn(self.batch_size, self.sequence_length,
                             self.vocab_size)
    logits = logits.astype(np.float32)
    sequence_length = np.array([1, 2, 3, 4])
    targets = np.random.randint(0, self.vocab_size,
                                [self.batch_size, self.sequence_length])
    losses = seq2seq_losses.cross_entropy_sequence_loss(logits, targets,
                                                        sequence_length)

    with self.test_session() as sess:
      losses_ = sess.run(losses)

    # Make sure all losses not past the sequence length are > 0
    np.testing.assert_array_less(np.zeros_like(losses_[0, :1]), losses_[0, :1])
    np.testing.assert_array_less(np.zeros_like(losses_[1, :2]), losses_[1, :2])
    np.testing.assert_array_less(np.zeros_like(losses_[2, :3]), losses_[2, :3])

    # Make sure all losses past the sequence length are 0
    np.testing.assert_array_equal(losses_[0, 1:], np.zeros_like(losses_[0, 1:]))
    np.testing.assert_array_equal(losses_[1, 2:], np.zeros_like(losses_[1, 2:]))
    np.testing.assert_array_equal(losses_[2, 3:], np.zeros_like(losses_[2, 3:]))


if __name__ == "__main__":
  tf.test.main()
