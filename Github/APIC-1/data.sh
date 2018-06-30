#!/bin/bash
wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip
wget https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip
unzip -p source-archive.zip  word2vec/trunk/questions-words.txt > questions-words.txt
rm text8.zip source-archive.zip
