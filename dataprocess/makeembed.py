from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle
import gensim


def get_embeddings(model, wrd_list, embed_dims):
    embed_list=[]
    # 添加OOV词向量0
    for word in wrd_list:
        if word in model.vocab:
            embed_list.append(model.word_vec(word))
        else:
            embed_list.append(np.zeros(embed_dims))
    return np.array(embed_list, dtype=np.float32)


# def make_embed(embed_loc, word_list, savepath):
#     model = gensim.models.KeyedVectors.load_word2vec_format(embed_loc, binary=False)
#     embed_init = get_embeddings(model, word_list, 50)
#     with tf.gfile.GFile(savepath, 'w') as f:
#         pickle.dump(embed_init, f, -1)


if __name__ == '__main__':
    embed_loc = '/data/MLRE-NG-archive/glove/glove.6B.50d_word2vec.txt'
    data = pickle.load(open('/data/MLRE-NG/PKL/dict.pkl', 'rb'))
    embed_path = '/data/MLRE-NG/embeddings.pkl'
    voc_path = '/data/MLRE-NG/vocab.pkl'
    voc2id = data['voc2id']

    # get word list
    word_list = list(voc2id.items())
    word_list.sort(key=lambda x: x[1])
    word_list, _ = zip(*word_list)
    with open(voc_path, 'wb') as f:
        pickle.dump(word_list, f)
    print(len(word_list))
    model = gensim.models.KeyedVectors.load_word2vec_format(embed_loc, binary=False)
    embed_init = get_embeddings(model, word_list, 50)
    print('embed size:')
    print(len(embed_init))
    with open(embed_path, 'wb') as f:
        pickle.dump(embed_init, f)
