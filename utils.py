import tensorflow as tf
import numpy as np
import os, sys, json, random, argparse
import logging, logging.config, pathlib

try:
    import cPickle as pickle
except:
    import pickle
import uuid, time, pdb, gensim, itertools
from collections import defaultdict as ddict
from pprint import pprint

# 所有文件的库都在这里导入

# 设置numpy的精度
np.set_printoptions(precision=4)


# 从pkl文件中读出embeddings
def load_pickle(path):
    with tf.gfile.GFile(path, 'rb') as f:
        return pickle.load(f)


# 检查路径下文件是否存在
def checkFile(filename):
    return pathlib.Path(filename).is_file()


# 创建路径
def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# 将输入列表分段为相同长度的多个列表
def getChunks(inp_list, chunk_size):
    return [inp_list[x:x + chunk_size] for x in range(0, len(inp_list), chunk_size)]


# gpu设置
def set_gpu(gpus):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus