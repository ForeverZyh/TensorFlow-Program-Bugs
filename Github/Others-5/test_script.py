import tensorflow as tf
import sys
import os
import subprocess

try:
    assert len(sys.argv) == 2
    version = ["-buggy", "-fix"][int(sys.argv[1])]
except:
    print(
        "Please run 'python test_script 0' for testing the buggy-version and "
        "'python test_script 1' for testing the fix-version.\nAborted...")
    exit(1)

interpreter_path = sys.executable
print("Running at: ", interpreter_path)

assert tf.__version__ == "1.8.0"


def get_target_dir():
    for x in os.listdir(os.path.dirname(os.path.abspath(__file__))):
        if version in x:
            return x
    raise ValueError("No dir ends with %s!" % version)


os.system(
    ' '.join([interpreter_path, "./%s/adversarial_text/data/gen_vocab.py" % get_target_dir(), "--output_dir=/tmp/IMDB",
              "--dataset=imdb", "--imdb_input_dir=/tmp/aclImdb", "--lowercase=False"]))
os.system(
    ' '.join([interpreter_path, "./%s/adversarial_text/data/gen_data.py" % get_target_dir(), "--output_dir=/tmp/IMDB",
              "--dataset=imdb", "--imdb_input_dir=/tmp/aclImdb", "--lowercase=False", "--label_gain=False"]))
subprocess.call(
    [interpreter_path, "./%s/adversarial_text/train_classifier.py" % get_target_dir(), "--num_classes=3",
     "--adv_training_method=vat"])
