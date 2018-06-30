Reference: [Sequence-to-Sequence with Attention Model for Text Summarization.](https://github.com/tensorflow/models/blob/master/research/textsum/README.md)

Note: This buggy program will go wrong only with **Tensorflow 1.0**.

Authors:

Xin Pan
Peter Liu (peterjliu@google.com, github:peterjliu)

<b>Dataset</b>

We used the Gigaword dataset described in [Rush et al. A Neural Attention Model
for Sentence Summarization](https://arxiv.org/abs/1509.00685).

We cannot provide the dataset due to the license. See ExampleGen in data.py
about the data format. data/data contains a toy example. Also see data/vocab
for example vocabulary format. In <b>How To Run</b> below, users can use toy
data and vocab provided in the data/ directory to run the training by replacing
the data directory flag.

data_convert_example.py contains example of convert between binary and text.

<b>Note</b>

The original program is intended for python2,
I modified to make it runnable under python3,
according to [this discussion](https://github.com/tensorflow/models/issues/4036)

<b>How To Run</b>

Prerequisite: install TensorFlow and Bazel.

```shell
$ cd models-fix     # or models-buggy

$ ls -R
.:
data  textsum  WORKSPACE

./data:
vocab data

./textsum:
batch_reader.py       beam_search.py       BUILD    README.md                    seq2seq_attention_model.py  data
data.py  seq2seq_attention_decode.py  seq2seq_attention.py        seq2seq_lib.py

./textsum/data:
data  vocab

$ bazel build -c opt --config=cuda textsum/...

# Run the training.
$ bazel-bin/textsum/seq2seq_attention \
    --mode=train \
    --article_key=article \
    --abstract_key=abstract \
    --data_path=data/data \
    --vocab_path=data/vocab \
    --log_root=textsum/log_root \
    --train_dir=textsum/log_root/train

### The following eval and decode part is not necessary for our test.
```
