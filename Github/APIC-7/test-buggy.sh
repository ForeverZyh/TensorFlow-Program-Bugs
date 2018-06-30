#!/bin/bash
cd models-buggy/ptb
python ptb_word_lm.py --data_path=../../simple-examples/data/ --model=small
