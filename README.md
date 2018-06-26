# TensorFlow-Program-Bugs

We're still working on automatically testing scripts for these bugs instead of manually testing, so test cases and scripts will be added later.

All reproducible bugs can be found in two files `StackOverflow` and `Github`. Every subject in these two files is named as `root_cause-id`, such as "APIC-1". File `root_cause-id` contains a buggy-version and a fix-version of the subject.

## Environment

Most of the sujects run on TensorFlow 1.8.0. The rest of them can only run on other specific versions of TensorFlow. 

Please make sure you get the following versions of TensorFlow:

* 1.8.0
* 1.0.0
* 0.12.1rc
* 0.7.0
* 0.5.0

## StackOverflow

The assertion in each subject describes the version of TensorFlow on which the subject was reproduced.

Inputs of most sujects are integrated in the programs. The rest of them have been downloaded. We have fixed the random seeds of `tensorflow`, `numpy`, and `random` to get more stable results.

## Github

To be continued...

##Citation

Paper describing the project:

Yuhao Zhang, Yifan Chen, Shing-Chi Cheung, Yingfei Xiong, Lu Zhang. An Empirical Study on TensorFlow Program Bugs. ISSTA 2018