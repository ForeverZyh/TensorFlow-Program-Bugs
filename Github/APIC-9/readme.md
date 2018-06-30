Reference: comments in the [source file](models-fix/translate/translate.py)

Running `python3 models-{buggy,fix}/translate/translate.py` will download the WMT corpus into `/tmp` and tokenize it in a very basic way,
and then start training a model saving checkpoints to `/tmp`.
