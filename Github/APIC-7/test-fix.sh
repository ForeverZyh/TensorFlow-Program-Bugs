#!/bin/bash
cd models-fix/ptb
python ptb_word_lm.py --data_path=../../simple-examples/data/ --model=small
