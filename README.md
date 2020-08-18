# Multi-Task Learning for Relation Extraction
This repository contains code for [*Multi-Task Learning for Relation Extraction*](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=8995371) in *Proceedings of 2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI)*. 
![](https://github.com/voidiak/MTRE/blob/master/figures/readme.jpg)
## Requirements
* [Tensorflow](https://www.tensorflow.org/)
* [Numpy](https://www.numpy.org)
* [Tensorpack](https://github.com/tensorpack/tensorpack)

This code has been run with TensorFlow 1.14, TensorPack 0.9.7 and Numpy 1.16.3; other versions may work, but have not been tested.

## Fetching and Preprocessing Data
See *workflow.pdf* for detail. We use [Riedel 2010 dataset](http://iesl.cs.umass.edu/riedel/ecml/) for evaluation. For part of dependency and entity type labels, we thank [*RESIDE*](https://github.com/malllabiisc/RESIDE) for providing processed data on their github page. For other missing labels, we use [StanfordNLP](https://stanfordnlp.github.io/stanfordnlp/) to obtain dependency labels and [FIGER](https://github.com/xiaoling/figer) to obtain entity type labels.

## Training and Evaluating a Model
Run `python edr.py pretrain` for pretraining, `python edr.py train` for training and `python edr.py eval` for evaluation.
