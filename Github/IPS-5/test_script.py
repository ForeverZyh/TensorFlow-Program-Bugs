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


subprocess.call(
    [interpreter_path, "./%s/command_line/run_rbm.py" % get_target_dir(), "--verbose=1", "--num_hidden=250", "--num_epochs=10",
     "--batch_size=128", "--model_name=rbm", "--learning_rate=0.0001"])
