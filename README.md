# TensorFlow-Program-Bugs

All reproducible bugs can be found in two files `StackOverflow` and `Github`. Every subject in these two files is named as `root_cause-id`, such as "APIC-1". File `root_cause-id` contains a buggy-version and a fix-version of the subject.

## Environment

Most of the sujects run on TensorFlow 1.8.0. The rest of them can only run on other specific versions of TensorFlow. 

We encoruage you to use `conda` or `virtualenv` to control your different versions of TensorFlow. Please make sure you get the following versions of TensorFlow:

* 1.8.0
* 1.1.0
* 1.0.0
* 0.12.0rc1
* 0.7.0
* 0.5.0

And we use python3 in most of the cases, except for `Github/APIM-4` using python2.7.x.

## StackOverflow

The assertion in each subject describes the version of TensorFlow on which the subject was reproduced.

Inputs of most subjects are integrated into the programs. The rest of them have been downloaded. We have fixed the random seeds of `tensorflow`, `numpy`, and `random` to get more stable results.

Except for `SI-1` and `Others-4` , all subjects contain only one entry python file which can be run using the command `python entry.py`. Testing methods of `SI-1` and `Others-4` are described in READMEs in these two files respectively.

## Github

We provide a `test_script.py` or `test-buggy.sh`,  `test-fix.sh` and `data.sh` as a test entry of most of sujects. The assertion in each `test_script.py` describes the version of TensorFlow and python on which the subject was reproduced.

We provide  `README`s for the rest of them requiring speacial build configuration.

## Datasets

We provide some useful datasets in the `Datasets` file. Some of them are partially downloaded since they are so large.

http://yann.lecun.com/exdb/mnist

http://vis-www.cs.umass.edu/lfw

http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

https://github.com/daviddao/spatial-transformer-tensorflow

http://www.robots.ox.ac.uk/~vgg/data/vgg_face2

http://ai.stanford.edu/~amaas/data/sentiment

## Citation

Paper describing the project:

Yuhao Zhang, Yifan Chen, Shing-Chi Cheung, Yingfei Xiong, Lu Zhang. An Empirical Study on TensorFlow Program Bugs. ISSTA 2018

@inproceedings{DBLP:conf/issta/ZhangCCXZ18,
  author    = {Yuhao Zhang and
               Yifan Chen and
               Shing{-}Chi Cheung and
               Yingfei Xiong and
               Lu Zhang},
  title     = {An empirical study on TensorFlow program bugs},
  booktitle = {Proceedings of the 27th {ACM} {SIGSOFT} International Symposium on
               Software Testing and Analysis, {ISSTA} 2018, Amsterdam, The Netherlands,
               July 16-21, 2018},
  pages     = {129--140},
  year      = {2018},
  crossref  = {DBLP:conf/issta/2018},
  url       = {https://doi.org/10.1145/3213846.3213866},
  doi       = {10.1145/3213846.3213866},
  timestamp = {Tue, 07 Jan 2020 08:30:57 +0100},
  biburl    = {https://dblp.org/rec/conf/issta/ZhangCCXZ18.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
