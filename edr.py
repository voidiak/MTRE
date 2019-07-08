from utils import *
from six.moves import range
from tensorpack import *
from tensorpack.tfutils.gradproc import GlobalNormClip, SummaryGradient
from tensorpack import ProxyDataFlow
from tensorpack.dataflow import LMDBSerializer, MultiProcessRunnerZMQ
from tensorpack.tfutils import optimizer
from tensorpack.utils import logger
from sklearn.metrics import precision_recall_curve, average_precision_score
import seaborn
import matplotlib
import gensim
# matplotlib.use('agg')
import matplotlib.pyplot as plt

plt.style.use('ggplot')

WORD_EMBED_DIM = 50
POS_EMBED_DIM = 5
ENTITY_TYPE_CLASS = 107
RELATION_TYPE_CLASS = 53
MAX_POS = (60 + 1) * 2 + 1
EMBED_LOC = '/data/MTRE-archive/glove/glove.6B.50d_word2vec.txt'
BASELINE_LOC = './baseline/'
VOCAB_LOC = './vocab.pkl'


class getbatch(ProxyDataFlow):

    def __init__(self, ds, batch, isTrain):
        self.batch = batch
        self.ds = ds
        self.isTrain = isTrain

    def __len__(self):
        return len(self.ds) // self.batch

    def __iter__(self):
        itr = self.ds.__iter__()
        for _ in range(self.__len__()):
            Xs, Pos1s, Pos2s, DepMasks, HeadPoss, TailPoss = [], [], [], [], [], []
            Y = []
            DepLabels, ReLabels, HeadLabels, TailLabels = [], [], [], []
            SentNum = []
            num = 0

            for b in range(self.batch):
                X, Pos1, Pos2, DepMask, HeadPos, TailPos, DepLabel, ReLabel, HeadLabel, TailLabel = next(itr)
                Xs += X
                Pos1s += Pos1
                Pos2s += Pos2
                DepMasks += DepMask
                HeadPoss += HeadPos
                TailPoss += TailPos
                DepLabels += DepLabel
                Y.append(ReLabel)

                HeadLabels.append(HeadLabel)
                TailLabels.append(TailLabel)
                old_num = num
                num += len(X)
                SentNum.append([old_num, num, b])

            Xs, X_len, Pos1s, Pos2s, DepMasks, DepLabels, max_seq_len = self.pad_dynamic(Xs, Pos1s, Pos2s, DepMasks,
                                                                                         DepLabels)
            ReLabels = self.getOneHot(Y, 53)
            total_sents = num
            total_bags = len(Y)
            if not self.isTrain:
                dropout = 1.0
                rec_dropout = 1.0
            else:
                dropout = 0.8
                rec_dropout = 0.8
            yield [Xs, Pos1s, Pos2s, HeadPoss, TailPoss, DepMasks, X_len, max_seq_len, total_sents, total_bags, SentNum,
                   ReLabels, DepLabels, HeadLabels, TailLabels, rec_dropout, dropout]

    def getOneHot(self, Y, re_num_class):
        temp = np.zeros((len(Y), re_num_class), np.int32)
        for i, e in enumerate(Y):
            for rel in e:
                temp[i, rel] = 1
        return temp

    def pad_dynamic(self, X, pos1, pos2, dep_mask, dep):
        # 为每个batch中的句子补位
        seq_len = 0
        x_len = np.zeros((len(X)), np.int32)

        for i, x in enumerate(X):
            seq_len = max(seq_len, len(x))
            x_len[i] = len(x)

        x_pad = self.padData(X, seq_len)
        pos1_pad = self.padData(pos1, seq_len)
        pos2_pad = self.padData(pos2, seq_len)
        dep_mask_pad = self.padData(dep_mask, seq_len)
        dep_pad = self.padData(dep, seq_len)

        return x_pad, x_len, pos1_pad, pos2_pad, dep_mask_pad, dep_pad, seq_len

    def padData(self, data, seq_len):
        # 为句子补位
        temp = np.zeros((len(data), seq_len), np.int32)

        for i, ele in enumerate(data):
            temp[i, :len(ele)] = ele[:seq_len]

        return temp


class WarmupModel(ModelDesc):
    def __init__(self, params):
        self.rnn_dim = params.rnn_dim
        self.proj_dim = params.proj_dim
        self.dep_proj_dim = params.dep_proj_dim
        self.lr = params.lr
        if params.l2 == 0.0:
            self.regularizer = None
        else:
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=params.l2)
        # self.embed_matrix = pickle.load(open(EMBED_LOC, 'rb'))
        self.vocab = pickle.load(open(VOCAB_LOC, 'rb'))

    def inputs(self):
        return [tf.TensorSpec([None, None], tf.int32, 'input_x'),  # Xs
                tf.TensorSpec([None, None], tf.int32, 'input_pos1'),  # Pos1s
                tf.TensorSpec([None, None], tf.int32, 'input_pos2'),  # Pos2s
                tf.TensorSpec([None], tf.int32, 'head_pos'),  # HeadPoss
                tf.TensorSpec([None], tf.int32, 'tail_pos'),  # TailPoss
                tf.TensorSpec([None, None], tf.float32, 'dep_mask'),  # DepMasks
                tf.TensorSpec([None], tf.int32, 'x_len'),  # X_len
                tf.TensorSpec((), tf.int32, 'seq_len'),  # max_seq_len
                tf.TensorSpec((), tf.int32, 'total_sents'),  # total_sents
                tf.TensorSpec((), tf.int32, 'total_bags'),  # total_bags
                tf.TensorSpec([None, 3], tf.int32, 'sent_num'),  # SentNum
                tf.TensorSpec([None, None], tf.int32, 'input_y'),  # ReLabels
                tf.TensorSpec([None, None], tf.int32, 'dep_y'),  # DepLabels
                tf.TensorSpec([None, None], tf.float32, 'head_label'),  # HeadLabels
                tf.TensorSpec([None, None], tf.float32, 'tail_label'),  # TailLabels
                tf.TensorSpec((), tf.float32, 'rec_dropout'),
                tf.TensorSpec((), tf.float32, 'dropout')
                ]

    def build_graph(self, input_x, input_pos1, input_pos2, head_pos, tail_pos, dep_mask, x_len, seq_len, total_sents,
                    total_bags, sent_num, input_y, dep_y, head_label, tail_label, rec_dropout, dropout):
        with tf.variable_scope('word_embedding'):
            model = gensim.models.KeyedVectors.load_word2vec_format(EMBED_LOC, binary=False)
            embed_init = get_embeddings(model, self.vocab, WORD_EMBED_DIM)
            _word_embeddings = tf.get_variable('embeddings', initializer=embed_init, trainable=True,
                                               regularizer=self.regularizer)
            # OOV pad
            zero_pad = tf.zeros([1, WORD_EMBED_DIM])
            word_embeddings = tf.concat([zero_pad, _word_embeddings], axis=0)
            pos1_embeddings = tf.get_variable('pos1_embeddings', [MAX_POS, POS_EMBED_DIM],
                                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True,
                                              regularizer=self.regularizer)
            pos2_embeddings = tf.get_variable('pos2_embeddings', [MAX_POS, POS_EMBED_DIM],
                                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True,
                                              regularizer=self.regularizer)

            word_embeded = tf.nn.embedding_lookup(word_embeddings, input_x)
            pos1_embeded = tf.nn.embedding_lookup(pos1_embeddings, input_pos1)
            pos2_embeded = tf.nn.embedding_lookup(pos2_embeddings, input_pos2)
            embeds = tf.concat([word_embeded, pos1_embeded, pos2_embeded], axis=2)

        with tf.variable_scope('Bi_rnn'):
            fw_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.rnn_dim, name='FW_GRU'),
                                                    output_keep_prob=rec_dropout)
            bk_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.rnn_dim, name='BW_GRU'),
                                                    output_keep_prob=rec_dropout)
            val, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bk_cell, embeds, sequence_length=x_len,
                                                         dtype=tf.float32)
            hidden_states = tf.concat((val[0], val[1]), axis=2)
            rnn_output_dim = self.rnn_dim * 2

        with tf.variable_scope('entity_type_classification'):
            entity_query = tf.get_variable('head_query', [rnn_output_dim, 1],
                                           initializer=tf.contrib.layers.xavier_initializer())
            # 以句子中的词index建立索引
            s_idx = tf.range(0, total_sents, 1, dtype=tf.int32)
            head_index = tf.concat(
                [tf.reshape(s_idx, [total_sents, 1]), tf.reshape(head_pos, [total_sents, 1])], axis=-1)
            tail_index = tf.concat(
                [tf.reshape(s_idx, [total_sents, 1]), tf.reshape(tail_pos, [total_sents, 1])], axis=-1)
            # add null word vector
            word_hidden_states = tf.concat([tf.zeros([total_sents, 1, rnn_output_dim]), hidden_states], axis=1)
            # extract head/tail entity's hidden state. size (total_sents,hidden_dim)
            head_repre_s = tf.gather_nd(word_hidden_states, head_index, name='head_entity_h_in_sentence')
            tail_repre_s = tf.gather_nd(word_hidden_states, tail_index, name='tail_entity_h_in_sentence')

            # 计算一个包中head实体的多个向量的att和
            def getHeadRepre(num):
                num_sents = num[1] - num[0]
                bag_sents = head_repre_s[num[0]:num[1]]

                head_att_weights = tf.nn.softmax(
                    tf.reshape(tf.matmul(tf.tanh(bag_sents), entity_query), [num_sents]))

                head_repre_ = tf.reshape(
                    tf.matmul(
                        tf.reshape(head_att_weights, [1, num_sents]),
                        bag_sents
                    ), [rnn_output_dim]
                )
                return head_repre_

            # 计算一个包中tail实体的多个向量的att和
            def getTailRepre(num):
                num_sents = num[1] - num[0]
                bag_sents = tail_repre_s[num[0]:num[1]]

                tail_att_weights = tf.nn.softmax(
                    tf.reshape(tf.matmul(tf.tanh(bag_sents), entity_query), [num_sents]))

                tail_repre_ = tf.reshape(
                    tf.matmul(
                        tf.reshape(tail_att_weights, [1, num_sents]),
                        bag_sents
                    ), [rnn_output_dim]
                )
                return tail_repre_

            # 一个batch中实体的向量表示dimension(batchsize,rnn_output_dim)
            head_repre_b = tf.map_fn(getHeadRepre, sent_num, dtype=tf.float32)
            tail_repre_b = tf.map_fn(getTailRepre, sent_num, dtype=tf.float32)

        with tf.variable_scope('entity_fully_connected_layer'):
            w_e = tf.get_variable('w', [rnn_output_dim, ENTITY_TYPE_CLASS],
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_e = tf.get_variable('b', initializer=np.zeros([ENTITY_TYPE_CLASS]).astype(np.float32))
            hr_out = tf.nn.xw_plus_b(head_repre_b, w_e, b_e)
            tr_out = tf.nn.xw_plus_b(tail_repre_b, w_e, b_e)

        # get ner accuracy
        ner_logits = tf.nn.softmax(hr_out)
        ner_pred = tf.argmax(ner_logits, axis=1)
        ner_actual = tf.argmax(head_label, axis=1)
        ner_accuracy_ = tf.cast(tf.equal(ner_pred, ner_actual), tf.float32, name='ner_accu')
        ner_accuracy = tf.reduce_mean(ner_accuracy_)

        with tf.variable_scope('dep_predictions'):
            arc_dep_hidden = tf.layers.dense(hidden_states, self.proj_dim, name='arc_dep_hidden')
            arc_head_hidden = tf.layers.dense(hidden_states, self.proj_dim, name='arc_head_hidden')
            # activation
            arc_dep_hidden = tf.nn.relu(arc_dep_hidden)
            arc_head_hidden = tf.nn.relu(arc_head_hidden)

            # dropout
            arc_dep_hidden = Dropout(arc_dep_hidden, keep_prob=dropout)
            arc_head_hidden = Dropout(arc_head_hidden, keep_prob=dropout)

            # bilinear classifier excluding the final dot product
            arc_head = tf.layers.dense(arc_head_hidden, self.dep_proj_dim, name='arc_head')
            W = tf.get_variable('shared_W', shape=[self.proj_dim, 1,
                                                   self.dep_proj_dim],
                                initializer=tf.contrib.layers.xavier_initializer())
            arc_dep = tf.tensordot(arc_dep_hidden, W, axes=[[-1], [0]])
            shape = tf.shape(arc_dep)
            arc_dep = tf.reshape(arc_dep, [shape[0], -1, self.dep_proj_dim])

            # apply the transformer trick to prevent dot products from getting too large
            scale = np.power(self.dep_proj_dim, 0.25).astype('float32')
            scale = tf.get_variable('scale', initializer=scale, dtype=tf.float32)
            arc_dep /= scale
            arc_head /= scale

            # compute the scores for each candidate arc
            word_score = tf.matmul(arc_head, arc_dep, transpose_b=True)
            arc_scores = word_score

            # disallow the model from making impossible predictions
            mask_shape = tf.shape(dep_mask)
            dep_mask_ = tf.tile(tf.expand_dims(dep_mask, 1), [1, mask_shape[1], 1])
            arc_scores += (dep_mask_ - 1) * 100
            nn_dep_out = arc_scores

        dep_labels = tf.one_hot(dep_y, seq_len, axis=-1, dtype=tf.int32, name='dep_label')
        # get dep accuracy
        dep_logits = tf.nn.softmax(nn_dep_out)
        dep_pred = tf.reshape(tf.argmax(dep_logits, axis=-1), [-1])
        dep_actual = tf.reshape(tf.argmax(dep_labels, axis=-1), [-1])
        y_mask = tf.cast(tf.reshape(dep_mask, [-1]), dtype=tf.bool)
        pred_masked = tf.boolean_mask(dep_pred, y_mask)
        actual_masked = tf.boolean_mask(dep_actual, y_mask)
        dep_accuracy_ = tf.cast(tf.equal(pred_masked, actual_masked), tf.float32, name='dep_accu')
        dep_accuracy = tf.reduce_mean(dep_accuracy_)
        # use sigmoid loss multi-label classification
        head_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hr_out, labels=head_label))
        tail_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tr_out, labels=tail_label))
        dep_ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=nn_dep_out, labels=dep_labels)
        dp_loss = tf.reduce_sum(dep_mask * dep_ce) / tf.to_float(tf.reduce_sum(dep_mask))
        loss = 0.3 * dp_loss + 0.35 * head_loss + 0.35 * tail_loss
        if self.regularizer is not None:
            loss += tf.contrib.layers.apply_regularization(self.regularizer,
                                                           tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = tf.identity(loss, name='total_loss')
        summary.add_moving_summary(loss, ner_accuracy, dep_accuracy)
        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=self.lr, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        return optimizer.apply_grad_processors(opt, [GlobalNormClip(5)])


class Model(ModelDesc):
    def __init__(self, params):
        self.rnn_dim = params.rnn_dim
        self.proj_dim = params.proj_dim
        self.dep_proj_dim = params.dep_proj_dim
        self.lr = params.lr
        if params.l2 == 0.0:
            self.regularizer = None
        else:
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=params.l2)
        # self.embed_matrix = pickle.load(open(EMBED_LOC, 'rb'))
        self.vocab = pickle.load(open(VOCAB_LOC, 'rb'))
        self.gcn_layers = 2
        self.gcn_dim = params.gcn_dim
        self.coe = params.coe

    def inputs(self):
        return [tf.TensorSpec([None, None], tf.int32, 'input_x'),  # Xs
                tf.TensorSpec([None, None], tf.int32, 'input_pos1'),  # Pos1s
                tf.TensorSpec([None, None], tf.int32, 'input_pos2'),  # Pos2s
                tf.TensorSpec([None], tf.int32, 'head_pos'),  # HeadPoss
                tf.TensorSpec([None], tf.int32, 'tail_pos'),  # TailPoss
                tf.TensorSpec([None, None], tf.float32, 'dep_mask'),  # DepMasks
                tf.TensorSpec([None], tf.int32, 'x_len'),  # X_len
                tf.TensorSpec((), tf.int32, 'seq_len'),  # max_seq_len
                tf.TensorSpec((), tf.int32, 'total_sents'),  # total_sents
                tf.TensorSpec((), tf.int32, 'total_bags'),  # total_bags
                tf.TensorSpec([None, 3], tf.int32, 'sent_num'),  # SentNum
                tf.TensorSpec([None, None], tf.int32, 'input_y'),  # ReLabels
                tf.TensorSpec([None, None], tf.int32, 'dep_y'),  # DepLabels
                tf.TensorSpec([None, None], tf.float32, 'head_label'),  # HeadLabels
                tf.TensorSpec([None, None], tf.float32, 'tail_label'),  # TailLabels
                tf.TensorSpec((), tf.float32, 'rec_dropout'),
                tf.TensorSpec((), tf.float32, 'dropout')
                ]

    def build_graph(self, input_x, input_pos1, input_pos2, head_pos, tail_pos, dep_mask, x_len, seq_len, total_sents,
                    total_bags, sent_num, input_y, dep_y, head_label, tail_label, rec_dropout, dropout):

        with tf.variable_scope('word_embedding'):
            model = gensim.models.KeyedVectors.load_word2vec_format(EMBED_LOC, binary=False)
            embed_init = get_embeddings(model, self.vocab, WORD_EMBED_DIM)
            _word_embeddings = tf.get_variable('embeddings', initializer=embed_init, trainable=True,
                                               regularizer=self.regularizer)
            # OOV pad
            zero_pad = tf.zeros([1, WORD_EMBED_DIM])
            word_embeddings = tf.concat([zero_pad, _word_embeddings], axis=0)
            pos1_embeddings = tf.get_variable('pos1_embeddings', [MAX_POS, POS_EMBED_DIM],
                                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True,
                                              regularizer=self.regularizer)
            pos2_embeddings = tf.get_variable('pos2_embeddings', [MAX_POS, POS_EMBED_DIM],
                                              initializer=tf.contrib.layers.xavier_initializer(), trainable=True,
                                              regularizer=self.regularizer)

            word_embeded = tf.nn.embedding_lookup(word_embeddings, input_x)
            pos1_embeded = tf.nn.embedding_lookup(pos1_embeddings, input_pos1)
            pos2_embeded = tf.nn.embedding_lookup(pos2_embeddings, input_pos2)
            embeds = tf.concat([word_embeded, pos1_embeded, pos2_embeded], axis=2)

        with tf.variable_scope('Bi_rnn'):
            fw_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.rnn_dim, name='FW_GRU'),
                                                    output_keep_prob=rec_dropout)
            bk_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.rnn_dim, name='BW_GRU'),
                                                    output_keep_prob=rec_dropout)
            val, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bk_cell, embeds, sequence_length=x_len,
                                                         dtype=tf.float32)
            hidden_states = tf.concat((val[0], val[1]), axis=2)
            rnn_output_dim = self.rnn_dim * 2

        with tf.variable_scope('entity_type_classification'):
            entity_query = tf.get_variable('entity_query', [rnn_output_dim, 1],
                                           initializer=tf.contrib.layers.xavier_initializer())
            # 以句子中的词index建立索引
            s_idx = tf.range(0, total_sents, 1, dtype=tf.int32)
            head_index = tf.concat(
                [tf.reshape(s_idx, [total_sents, 1]), tf.reshape(head_pos, [total_sents, 1])], axis=-1)
            tail_index = tf.concat(
                [tf.reshape(s_idx, [total_sents, 1]), tf.reshape(tail_pos, [total_sents, 1])], axis=-1)
            # add null word vector
            word_hidden_states = tf.concat([tf.zeros([total_sents, 1, rnn_output_dim]), hidden_states], axis=1)
            # extract head/tail entity's hidden state. size (total_sents,hidden_dim)
            head_repre_s = tf.gather_nd(word_hidden_states, head_index, name='head_entity_h_in_sentence')
            tail_repre_s = tf.gather_nd(word_hidden_states, tail_index, name='tail_entity_h_in_sentence')

            # 计算一个包中head实体的多个向量的att和
            def getHeadRepre(num):
                num_sents = num[1] - num[0]
                bag_sents = head_repre_s[num[0]:num[1]]

                head_att_weights = tf.nn.softmax(
                    tf.reshape(tf.matmul(tf.tanh(bag_sents), entity_query), [num_sents]))

                head_repre_ = tf.reshape(
                    tf.matmul(
                        tf.reshape(head_att_weights, [1, num_sents]),
                        bag_sents
                    ), [rnn_output_dim]
                )
                return head_repre_

            # 计算一个包中tail实体的多个向量的att和
            def getTailRepre(num):
                num_sents = num[1] - num[0]
                bag_sents = tail_repre_s[num[0]:num[1]]

                tail_att_weights = tf.nn.softmax(
                    tf.reshape(tf.matmul(tf.tanh(bag_sents), entity_query), [num_sents]))

                tail_repre_ = tf.reshape(
                    tf.matmul(
                        tf.reshape(tail_att_weights, [1, num_sents]),
                        bag_sents
                    ), [rnn_output_dim]
                )
                return tail_repre_

            # 一个batch中实体的向量表示 dimension(batchsize,rnn_output_dim)
            head_repre_b = tf.map_fn(getHeadRepre, sent_num, dtype=tf.float32)
            tail_repre_b = tf.map_fn(getTailRepre, sent_num, dtype=tf.float32)

        with tf.variable_scope('dep_predictions'):
            # Projection 考虑现在hidden states是多个句子的串联，用cnn
            arc_dep_hidden = tf.layers.dense(hidden_states, self.proj_dim, name='arc_dep_hidden')
            arc_head_hidden = tf.layers.dense(hidden_states, self.proj_dim, name='arc_head_hidden')

            # activation
            arc_dep_hidden = tf.nn.relu(arc_dep_hidden)
            arc_head_hidden = tf.nn.relu(arc_head_hidden)

            # dropout
            arc_dep_hidden = Dropout(arc_dep_hidden, keep_prob=dropout)
            arc_head_hidden = Dropout(arc_head_hidden, keep_prob=dropout)

            # bilinear classifier excluding the final dot product
            arc_head = tf.layers.dense(arc_head_hidden, self.dep_proj_dim, name='arc_head')
            W = tf.get_variable('shared_W', shape=[self.proj_dim, 1,
                                                   self.dep_proj_dim])
            arc_dep = tf.tensordot(arc_dep_hidden, W, axes=[[-1], [0]])
            shape = tf.shape(arc_dep)
            arc_dep = tf.reshape(arc_dep, [shape[0], -1, self.dep_proj_dim])

            # apply the transformer trick to prevent dot products from getting too large
            scale = np.power(self.dep_proj_dim, 0.25).astype('float32')
            scale = tf.get_variable('scale', initializer=scale, dtype=tf.float32)
            arc_dep /= scale
            arc_head /= scale

            # compute the scores for each candidate arc
            word_score = tf.matmul(arc_head, arc_dep, transpose_b=True)
            arc_scores = word_score

        # gcn encoding dependency tree structure
        dep_matrix = tf.nn.softmax(arc_scores)
        gcn_matrix = tf.transpose(dep_matrix, [0, 2, 1])
        #warning
        gcn_matrix = gcn_matrix + tf.eye(seq_len)

        with tf.variable_scope('gcn_encoder') as scope:
            denom = tf.expand_dims(tf.reduce_sum(gcn_matrix, axis=2), axis=2) + 1
            # gcn_mask = tf.expand_dims(
            #     tf.equal((tf.reduce_sum(dep_matrix, axis=2) + tf.reduce_sum(dep_matrix, axis=1)), 0), axis=2)
            for l in range(self.gcn_layers):
                Ax = tf.matmul(gcn_matrix, hidden_states)
                AxW = tf.layers.dense(Ax, self.gcn_dim)
                AxW = AxW + tf.layers.dense(hidden_states, self.gcn_dim)
                AxW = AxW / denom
                gAxW = tf.nn.relu(AxW)
                hidden_states = Dropout(gAxW, keep_prob=0.5) if l < self.gcn_layers - 1 else gAxW

        de_out_dim = self.gcn_dim

        # word attention
        with tf.variable_scope('word_attention') as scope:
            word_query = tf.get_variable('word_query', [de_out_dim, 1],
                                         initializer=tf.contrib.layers.xavier_initializer())
            sent_repre = tf.reshape(
                tf.matmul(
                    tf.reshape(
                        tf.nn.softmax(
                            tf.reshape(
                                tf.matmul(
                                    tf.reshape(tf.tanh(hidden_states),
                                               [total_sents * seq_len, de_out_dim]),
                                    word_query
                                ), [total_sents, seq_len]
                            )
                        ), [total_sents, 1, seq_len]
                    ), hidden_states
                ), [total_sents, de_out_dim]
            )

        # 包的表示

        with tf.variable_scope('sentence_attention') as scope:
            sentence_query = tf.get_variable('sentence_query', [de_out_dim, 1],
                                             initializer=tf.contrib.layers.xavier_initializer())

            def getSentenceAtt(num):
                num_sents = num[1] - num[0]
                bag_sents = sent_repre[num[0]:num[1]]

                sentence_att_weights = tf.nn.softmax(
                    tf.reshape(tf.matmul(tf.tanh(bag_sents), sentence_query), [num_sents]))

                bag_repre_ = tf.reshape(
                    tf.matmul(
                        tf.reshape(sentence_att_weights, [1, num_sents]),
                        bag_sents
                    ), [de_out_dim]
                )
                return bag_repre_

            bag_repre = tf.map_fn(getSentenceAtt, sent_num, dtype=tf.float32)

        bag_repre = tf.concat([bag_repre, head_repre_b, tail_repre_b], axis=-1)
        de_out_dim = de_out_dim + 4 * self.rnn_dim

        with tf.variable_scope('fully_connected_layer') as scope:
            w = tf.get_variable('w', [de_out_dim, RELATION_TYPE_CLASS],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', initializer=np.zeros([RELATION_TYPE_CLASS]).astype(np.float32))
            re_out = tf.nn.xw_plus_b(bag_repre, w, b)
            re_out = Dropout(re_out, keep_prob=dropout)

        re_logits = tf.nn.softmax(re_out, name='logits')
        re_pred = tf.argmax(re_logits, axis=1, name='pred_y')
        re_actual = tf.argmax(input_y, axis=1)
        re_accuracy_ = tf.cast(tf.equal(re_pred, re_actual), tf.float32, name='re_accu')
        re_accuracy = tf.reduce_mean(re_accuracy_)

        with tf.variable_scope('entity_fully_connected_layer') as scope:
            w_e = tf.get_variable('w', [rnn_output_dim, ENTITY_TYPE_CLASS],
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_e = tf.get_variable('b', initializer=np.zeros([ENTITY_TYPE_CLASS]).astype(np.float32))
            hr_out = tf.nn.xw_plus_b(head_repre_b, w_e, b_e)
            tr_out = tf.nn.xw_plus_b(tail_repre_b, w_e, b_e)

        label_y = tf.one_hot(dep_y, seq_len, axis=-1, dtype=tf.int32, name='dep_label')
        # use sigmoid loss multi-label classification
        head_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hr_out,
                                                                           labels=head_label))
        tail_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tr_out, labels=tail_label))

        dep_ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=arc_scores, labels=label_y)
        dp_loss = tf.reduce_sum(dep_mask * dep_ce) / tf.to_float(tf.reduce_sum(dep_mask))
        re_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=re_out, labels=input_y))
        loss = (1 - self.coe) * re_loss + self.coe * (0.35 * head_loss + 0.35 * tail_loss + 0.3 * dp_loss)
        if self.regularizer is not None:
            loss += tf.contrib.layers.apply_regularization(self.regularizer,
                                                           tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        loss = tf.identity(loss, name='total_loss')
        summary.add_moving_summary(loss, re_accuracy)
        return loss

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=self.lr, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        return optimizer.apply_grad_processors(opt, [GlobalNormClip(5)])


def getdata(path, batchsize, isTrain):
    ds = LMDBSerializer.load(path, shuffle=isTrain)
    ds = getbatch(ds, batchsize, isTrain)
    # if isTrain:
    #     ds = MultiProcessRunnerZMQ(ds, 2)
    return ds


def get_config(ds_train, ds_test, params):
    return TrainConfig(
        data=QueueInput(ds_train),
        callbacks=[
            ModelSaver(),
            StatMonitorParamSetter('learning_rate', 'total_loss',
                                   lambda x: x * 0.2, 0, 5),
            PeriodicTrigger(
                InferenceRunner(ds_test, [ScalarStats('total_loss'), ClassificationError('ner_accu', 'ner_accuracy'),
                                          ClassificationError('dep_accu', 'dep_accuracy')]),
                every_k_epochs=1),
            MovingAverageSummary(),
            MergeAllSummaries(),
        ],
        model=WarmupModel(params),
        max_epoch=params.pre_epochs,
    )


def resume_train(ds_train, ds_test, model_path, params, current_epoch, add_epochs):
    return AutoResumeTrainConfig(
        always_resume=False,
        data=QueueInput(ds_train),
        session_init=get_model_loader(model_path),
        starting_epoch=current_epoch + 1,
        callbacks=[
            ModelSaver(),
            StatMonitorParamSetter('learning_rate', 'total_loss',
                                   lambda x: x * 0.2, 0, 5),
            PeriodicTrigger(
                InferenceRunner(ds_test, [ScalarStats('total_loss'), ClassificationError('re_accu', 'accuracy')]),
                every_k_epochs=1),
            MovingAverageSummary(),
            MergeAllSummaries(),
            # GPUMemoryTracker(),
        ],
        model=Model(params),
        max_epoch=current_epoch + add_epochs,
    )


def evaluatepn(model, model_path, data_path, batchsize):
    ds = getdata(data_path, batchsize, False)
    eval_config = PredictConfig(
        model=model,
        session_init=get_model_loader(model_path),
        input_names=['input_x', 'input_pos1', 'input_pos2', 'head_pos', 'tail_pos', 'dep_mask', 'x_len', 'seq_len',
                     'total_sents', 'total_bags', 'sent_num', 'input_y', 'dep_y', 'head_label', 'tail_label',
                     'rec_dropout', 'dropout'],
        output_names=['logits', 'input_y']
    )
    pred = SimpleDatasetPredictor(eval_config, ds)

    logit_list, label_list, y_pred, y_gold = [], [], [], []

    for output in pred.get_result():
        logit_list += output[0].tolist()
        label_list += output[1].tolist()
        y_pred += output[0].argmax(axis=1).tolist()
        y_gold += output[1].argmax(axis=1).tolist()

    y_scores = np.array([e[1:] for e in logit_list]).reshape((-1))
    y_true = np.array([e[1:] for e in label_list]).reshape((-1))

    allprob = np.reshape(np.array(y_scores), (-1))
    allans = np.reshape(y_true, (-1))
    order = np.argsort(-allprob)

    def p_score(n):
        correct_num = 0.0
        for i in order[:n]:
            correct_num += 1.0 if (allans[i] == 1) else 0
        return correct_num / n

    return p_score(100), p_score(200), p_score(300)


def evaluate(model, model_path, data_path, batchsize):
    ds = getdata(data_path, batchsize, False)
    eval_config = PredictConfig(
        model=model,
        session_init=get_model_loader(model_path),
        input_names=['input_x', 'input_pos1', 'input_pos2', 'head_pos', 'tail_pos', 'dep_mask', 'x_len', 'seq_len',
                     'total_sents', 'total_bags', 'sent_num', 'input_y', 'dep_y', 'head_label', 'tail_label',
                     'rec_dropout', 'dropout'],
        output_names=['logits', 'input_y']
    )
    pred = SimpleDatasetPredictor(eval_config, ds)

    logit_list, label_list, y_pred, y_gold = [], [], [], []

    for output in pred.get_result():
        logit_list += output[0].tolist()
        label_list += output[1].tolist()
        y_pred += output[0].argmax(axis=1).tolist()
        y_gold += output[1].argmax(axis=1).tolist()

    y_scores = np.array([e[1:] for e in logit_list]).reshape((-1))
    y_true = np.array([e[1:] for e in label_list]).reshape((-1))

    precsion, recall, f1 = calculate_prf(y_gold, y_pred)
    area_under_pr = average_precision_score(y_true, y_scores)
    # precision_, recall_, threshold = precision_recall_curve(y_true, y_scores)
    precision_, recall_ ,p10_result= curve(y_scores, y_true, 2000)

    return precsion, recall, f1, area_under_pr, precision_, recall_, p10_result


def curve(y_scores, y_true, num=2000):
    order = np.argsort(y_scores)[::-1]
    guess = 0.
    right = 0.
    target = np.sum(y_true)
    precisions = []
    recalls = []
    for i in order[:num]:
        guess += 1
        if y_true[i] == 1:
            right += 1
        precision = right / guess
        recall = right / target
        precisions.append(precision)
        recalls.append(recall)

    ppp_num = len(y_scores) // 10
    ppp_result=0.0
    for i in order[:ppp_num]:
        # correct_num += 1.0 if (y_true[i] == 1) else 0
        ppp_result+=y_scores[i]
    ppp_result = ppp_result / ppp_num
    logger.info('P@10%:\t{}'.format(ppp_result))

    return np.array(precisions), np.array(recalls), ppp_result


def calculate_prf(gold, pred):
    pos_pred, pos_gt, true_pos = 0.0, 0.0, 0.0
    for i in range(len(gold)):
        if gold[i] != 0:
            pos_gt += 1.0
    for i in range(len(pred)):
        if pred[i] != 0:
            pos_pred += 1.0  # classified as pos example (Is-A-Relation)
            if pred[i] == gold[i]:
                true_pos += 1.0

    precision = true_pos / (pos_pred + 1e-8)
    recall = true_pos / (pos_gt + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def plotPRCurve(precision, recall, dir):
    plt.plot(recall[:], precision[:], label='MLRE', color='red', lw=1, marker='o', markevery=0.1, ms=6)

    base_list = ['BGWA', 'PCNN+ATT', 'PCNN', 'MIMLRE', 'MultiR', 'Mintz', 'RESIDE']
    color = ['purple', 'darkorange', 'green', 'xkcd:azure', 'orchid', 'cornflowerblue', 'yellow']
    marker = ['d', 's', '^', '*', 'v', 'x', 'h', 'p']
    plt.ylim([0.3, 1.0])
    plt.xlim([0.0, 0.45])

    for i, baseline in enumerate(base_list):
        precision_b = np.load(BASELINE_LOC + baseline + '/precision.npy')
        recall_b = np.load(BASELINE_LOC + baseline + '/recall.npy')
        plt.plot(recall_b, precision_b, color=color[i], label=baseline, lw=1, marker=marker[i], markevery=0.1, ms=6)

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.legend(loc="upper right", prop={'size': 12})
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plot_path = '{}/pr.pdf'.format(dir)
    plt.savefig(plot_path)
    print('Precision-Recall plot saved at: {}'.format(plot_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', dest='gpu', default='0', help='gpu to use')
    parser.add_argument('-l2', dest='l2', default=1e-4, type=float, help='l2 regularization')
    parser.add_argument('-seed', dest='seed', default=15, type=int, help='seed for randomization')
    parser.add_argument('-rnn_dim', dest='rnn_dim', default=180, type=int, help='hidden state dimension of Bi-RNN')
    parser.add_argument('-gcn_dim', dest='gcn_dim', default=360, type=int, help='hidden state dimension of GCN')
    parser.add_argument('-proj_dim', dest='proj_dim', default=256, type=int,
                        help='projection size for GRUs and hidden layers')
    parser.add_argument('-dep_proj_dim', dest='dep_proj_dim', default=64, type=int,
                        help='size of the representations used in the bilinear classifier for parsing')
    parser.add_argument('-coe', dest='coe', default=0.3, type=float, help='value for loss addition')
    parser.add_argument('-lr', dest='lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-pre_epochs', dest='pre_epochs', default=3, type=int, help='pretraining epochs')
    parser.add_argument('-epochs', dest='epochs', default=2, type=int, help='epochs to train/predict')
    parser.add_argument('-batch_size', dest='batch_size', default=200, type=int, help='batch size')
    subparsers = parser.add_subparsers(title='command', dest='command')
    parser_pretrain = subparsers.add_parser('pretrain')
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('-previous_model', dest='previous_model', default=0, type=int,
                              help='previous model to resume')
    parser_train.add_argument('-add_epochs', dest='add_epochs', default=0, type=int, help='epochs to continue')
    parser_evaluate = subparsers.add_parser('eval')
    parser_evaluate.add_argument('-best_model', dest='best_model', default=0, type=int, help='best model to evaluate')
    parser_evaluate.add_argument('-add_epochs', dest='add_epochs', default=0, type=int, help='epochs to continue')
    args = parser.parse_args()
    argdict = vars(args)
    name = 'seed_{}'.format(argdict['seed'])
    logger.auto_set_dir(action='k', name=name)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    step = int(293142 / args.batch_size)
    if args.command == 'pretrain':
        # set seed
        tf.set_random_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        # train
        ds = getdata('./mdb100/train.mdb', args.batch_size, True)
        dss = getdata('./mdb100/test.mdb', args.batch_size, False)
        config = get_config(ds, dss, args)
        launch_train_with_config(config, SimpleTrainer())
    elif args.command == 'train':
        ds = getdata('./mdb100/train.mdb', args.batch_size, True)
        dss = getdata('./mdb100/test.mdb', args.batch_size, False)
        # resume
        if args.previous_model:
            current_epoch = args.previous_model // step
            load_path = './train_log/edr:{}/model-{}'.format(name, args.previous_model)
            resume_config = resume_train(ds, dss, load_path, args, current_epoch, args.add_epochs)
            launch_train_with_config(resume_config, SimpleTrainer())
        else:
            current_step = step * args.pre_epochs
            load_path = './train_log/edr:{}/model-{}'.format(name, current_step)
            resume_config = resume_train(ds, dss, load_path, args, args.pre_epochs, args.epochs)
            launch_train_with_config(resume_config, SimpleTrainer())
    elif args.command == 'eval':
        # predict
        if args.best_model:
            test_path = './mdb100/test.mdb'
            best_model_path = os.path.join('./train_log/edr:{}/'.format(name), 'model-' + str(args.best_model))
            p, r, f1, aur, p_, r_ ,p10_result= evaluate(Model(args), best_model_path, test_path, args.batch_size)
            plotPRCurve(p_, r_, './train_log/edr:{}'.format(name))
            pickle.dump({'precision': p_, 'recall': r_}, open('./train_log/edr:{}/p_r.pkl'.format(name), 'wb'))
            with open('./train_log/edr:{}/{}.txt'.format(name, 'best_model'), 'w', encoding='utf-8')as f:
                f.write('precision:\t{}\nrecall:\t{}\nf1:\t{}\nauc:\t{}\n'.format(p, r, f1, aur))
                f.write(name + '\n')
                f.write('model name:' + str(args.best_model) + '\t' + '\n')
                for data in ['pn1', 'pn2', 'pn3', 'test_r', 'test']:
                    data_path = './mdb100/{}.mdb'.format(data)
                    p100, p200, p300 = evaluatepn(Model(args), best_model_path, data_path, args.batch_size)
                    logger.info('    {}:P@100:{:.3f}  P@200:{:.3f}  P@300:{:.3f}\n'.format(data, p100, p200, p300))
                    line = "{}:\t{:.3f}\t{:.3f}\t{:.3f}\t\n".format(data, p100, p200, p300)
                    f.write(line)
                f.write("p@10%:\t{}".format(p10_result))
                f.write('\n')
                f.close()
        else:
            with open('./train_log/edr:{}/{}.txt'.format(name, name), 'w', encoding='utf-8')as f:
                f.write(name + '\n')
                for model in [str(step * (args.pre_epochs + 1) + i * step) for i in
                              range(args.epochs + args.add_epochs)]:
                    f.write(model + '\t')
                    for data in ['pn1', 'pn2', 'test']:
                        data_path = './mdb100/{}.mdb'.format(data)
                        p100, p200, p300 = evaluatepn(Model(args), os.path.join('./train_log/edr:{}/'.format(name),
                                                                                'model-' + model), data_path,
                                                      args.batch_size)
                        logger.info('    {}:P@100:{:.3f}  P@200:{:.3f}  P@300:{:.3f}\n'.format(data, p100, p200, p300))
                        line = "{:.3f}\t{:.3f}\t{:.3f}\t".format(p100, p200, p300)
                        f.write(line)
                    f.write('\n')
                f.close()
