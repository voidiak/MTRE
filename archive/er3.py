from utils import *
from six.moves import range
from tensorpack import *
from tensorpack.tfutils.gradproc import GlobalNormClip, SummaryGradient
from tensorpack import ProxyDataFlow
from tensorpack.dataflow import LMDBSerializer, MultiProcessRunnerZMQ
from tensorpack.tfutils import optimizer
from tensorpack.utils import logger
import gensim


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
                X, Pos1, Pos2, DepMask, HeadPos, TailPos, DepLabel, ReLabel, HeadLabel, TailLabel = next(
                    itr
                )
                Xs += X
                Pos1s += Pos1
                Pos2s += Pos2
                HeadPoss += HeadPos
                TailPoss += TailPos
                Y.append(ReLabel)

                HeadLabels.append(HeadLabel)
                TailLabels.append(TailLabel)
                old_num = num
                num += len(X)
                SentNum.append([old_num, num, b])

            Xs, X_len, Pos1s, Pos2s, max_seq_len = self.pad_dynamic(Xs, Pos1s, Pos2s)
            Xs = np.array(Xs)
            ReLabels = self.getOneHot(Y, 53)
            total_sents = num
            total_bags = len(Y)
            if not self.isTrain:
                dropout = 1.0
                rec_dropout = 1.0
            else:
                dropout = 0.8
                rec_dropout = 0.8
            yield [
                Xs,
                Pos1s,
                Pos2s,
                HeadPoss,
                TailPoss,
                X_len,
                max_seq_len,
                total_sents,
                total_bags,
                SentNum,
                ReLabels,
                HeadLabels,
                TailLabels,
                rec_dropout,
                dropout,
            ]

    def getOneHot(self, Y, re_num_class):
        temp = np.zeros((len(Y), re_num_class), np.int32)
        for i, e in enumerate(Y):
            for rel in e:
                temp[i, rel] = 1
        return temp

    def pad_dynamic(self, X, pos1, pos2):
        # 为每个batch中的句子补位
        seq_len = 0
        x_len = np.zeros((len(X)), np.int32)

        for i, x in enumerate(X):
            seq_len = max(seq_len, len(x))
            x_len[i] = len(x)

        x_pad, _ = self.padData(X, seq_len)
        pos1_pad, _ = self.padData(pos1, seq_len)
        pos2_pad, _ = self.padData(pos2, seq_len)

        return x_pad, x_len, pos1_pad, pos2_pad, seq_len

    def padData(self, data, seq_len):
        # 为句子补位
        temp = np.zeros((len(data), seq_len), np.int32)
        mask = np.zeros((len(data), seq_len), np.float32)

        for i, ele in enumerate(data):
            temp[i, : len(ele)] = ele[:seq_len]
            mask[i, : len(ele)] = np.ones(len(ele[:seq_len]), np.float32)

        return temp, mask


class WarmupModel(ModelDesc):
    def __init__(self, params):
        self.params = params

        if self.params.l2 == 0.0:
            self.regularizer = None
        else:
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.params.l2)

        self.load_data()

    def load_data(self):
        data = pickle.load(open(self.params.dataset, "rb"))

        self.voc2id = data["voc2id"]
        self.id2voc = data["id2voc"]
        self.max_pos = data["max_pos"]
        self.num_class = len(data["rel2id"])
        self.num_entity_class = len(data["e_type2id"])
        self.num_deLabel = 1

        # get word list
        self.word_list = list(self.voc2id.items())
        self.word_list.sort(key=lambda x: x[1])
        self.word_list, _ = zip(*self.word_list)

    def inputs(self):
        return [
            tf.TensorSpec([None, None], tf.int32, "input_x"),  # Xs
            tf.TensorSpec([None, None], tf.int32, "input_pos1"),  # Pos1s
            tf.TensorSpec([None, None], tf.int32, "input_pos2"),  # Pos2s
            tf.TensorSpec([None], tf.int32, "head_pos"),  # HeadPoss
            tf.TensorSpec([None], tf.int32, "tail_pos"),  # TailPoss
            tf.TensorSpec([None], tf.int32, "x_len"),  # X_len
            tf.TensorSpec((), tf.int32, "seq_len"),  # max_seq_len
            tf.TensorSpec((), tf.int32, "total_sents"),  # total_sents
            tf.TensorSpec((), tf.int32, "total_bags"),  # total_bags
            tf.TensorSpec([None, 3], tf.int32, "sent_num"),  # SentNum
            tf.TensorSpec([None, None], tf.int32, "input_y"),  # ReLabels
            tf.TensorSpec([None, None], tf.float32, "head_label"),  # HeadLabels
            tf.TensorSpec([None, None], tf.float32, "tail_label"),  # TailLabels
            tf.TensorSpec((), tf.float32, "rec_dropout"),
            tf.TensorSpec((), tf.float32, "dropout"),
        ]

    def build_graph(
        self,
        input_x,
        input_pos1,
        input_pos2,
        head_pos,
        tail_pos,
        x_len,
        seq_len,
        total_sents,
        total_bags,
        sent_num,
        input_y,
        head_label,
        tail_label,
        rec_dropout,
        dropout,
    ):
        with tf.variable_scope("word_embedding") as scope:
            model = gensim.models.KeyedVectors.load_word2vec_format(
                self.params.embed_loc, binary=False
            )
            embed_init = getEmbeddings(
                model, self.word_list, self.params.word_embed_dim
            )
            _word_embeddings = tf.get_variable(
                "embeddings",
                initializer=embed_init,
                trainable=True,
                regularizer=self.regularizer,
            )
            word_pad = tf.zeros(
                [1, self.params.word_embed_dim]
            )  # word embedding for 'UNK'
            word_embeddings = tf.concat([word_pad, _word_embeddings], axis=0)

            pos1_embeddings = tf.get_variable(
                "pos1_embeddings",
                [self.max_pos, self.params.pos_dim],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True,
                regularizer=self.regularizer,
            )
            pos2_embeddings = tf.get_variable(
                "pos2_embeddings",
                [self.max_pos, self.params.pos_dim],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True,
                regularizer=self.regularizer,
            )

            word_embeded = tf.nn.embedding_lookup(word_embeddings, input_x)
            pos1_embeded = tf.nn.embedding_lookup(pos1_embeddings, input_pos1)
            pos2_embeded = tf.nn.embedding_lookup(pos2_embeddings, input_pos2)
            embeds = tf.concat([word_embeded, pos1_embeded, pos2_embeded], axis=2)

        with tf.variable_scope("Bi_rnn") as scope:
            fw_cell = tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(self.params.rnn_dim, name="FW_GRU"),
                output_keep_prob=rec_dropout,
            )
            bk_cell = tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(self.params.rnn_dim, name="BW_GRU"),
                output_keep_prob=rec_dropout,
            )
            val, state = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bk_cell, embeds, sequence_length=x_len, dtype=tf.float32
            )

            hidden_states = tf.concat((val[0], val[1]), axis=2)
            rnn_output_dim = self.params.rnn_dim * 2

        # word attention
        # with tf.variable_scope('word_attention') as scope:
        #     word_query = tf.get_variable('word_query', [rnn_output_dim, 1],
        #                                  initializer=tf.contrib.layers.xavier_initializer())
        #     sent_repre = tf.reshape(
        #         tf.matmul(
        #             tf.reshape(
        #                 tf.nn.softmax(
        #                     tf.reshape(
        #                         tf.matmul(
        #                             tf.reshape(tf.tanh(hidden_states),
        #                                        [total_sents * seq_len, rnn_output_dim]),
        #                             word_query
        #                         ), [total_sents, seq_len]
        #                     )
        #                 ), [total_sents, 1, seq_len]
        #             ), hidden_states
        #         ), [total_sents, rnn_output_dim]
        #     )
        # add entity type classification
        with tf.variable_scope("entity_type_classification"):
            entity_query = tf.get_variable(
                "head_query",
                [rnn_output_dim, 1],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            # 以句子中的词index建立索引
            s_idx = tf.range(0, total_sents, 1, dtype=tf.int32)
            head_index = tf.concat(
                [
                    tf.reshape(s_idx, [total_sents, 1]),
                    tf.reshape(head_pos, [total_sents, 1]),
                ],
                axis=-1,
            )
            tail_index = tf.concat(
                [
                    tf.reshape(s_idx, [total_sents, 1]),
                    tf.reshape(tail_pos, [total_sents, 1]),
                ],
                axis=-1,
            )
            # add null word vector
            word_hidden_states = tf.concat(
                [tf.zeros([total_sents, 1, rnn_output_dim]), hidden_states], axis=1
            )
            # extract head/tail entity's hidden state. size (total_sents,hidden_dim)
            head_repre_s = tf.gather_nd(
                word_hidden_states, head_index, name="head_entity_h_in_sentence"
            )
            tail_repre_s = tf.gather_nd(
                word_hidden_states, tail_index, name="tail_entity_h_in_sentence"
            )

            # 计算一个包中head实体的多个向量的att和
            def getHeadRepre(num):
                num_sents = num[1] - num[0]
                bag_sents = head_repre_s[num[0] : num[1]]

                head_att_weights = tf.nn.softmax(
                    tf.reshape(tf.matmul(tf.tanh(bag_sents), entity_query), [num_sents])
                )

                head_repre_ = tf.reshape(
                    tf.matmul(tf.reshape(head_att_weights, [1, num_sents]), bag_sents),
                    [rnn_output_dim],
                )
                return head_repre_

            # 计算一个包中tail实体的多个向量的att和
            def getTailRepre(num):
                num_sents = num[1] - num[0]
                bag_sents = tail_repre_s[num[0] : num[1]]

                tail_att_weights = tf.nn.softmax(
                    tf.reshape(tf.matmul(tf.tanh(bag_sents), entity_query), [num_sents])
                )

                tail_repre_ = tf.reshape(
                    tf.matmul(tf.reshape(tail_att_weights, [1, num_sents]), bag_sents),
                    [rnn_output_dim],
                )
                return tail_repre_

            # 一个batch中实体的向量表示
            head_repre_b = tf.map_fn(
                getHeadRepre, sent_num, dtype=tf.float32
            )  # dimension(batchsize,rnn_output_dim)
            tail_repre_b = tf.map_fn(getTailRepre, sent_num, dtype=tf.float32)

        with tf.variable_scope("entity_fully_connected_layer") as scope:
            w_e = tf.get_variable(
                "w",
                [rnn_output_dim, self.num_entity_class],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=self.regularizer,
            )
            b_e = tf.get_variable(
                "b",
                initializer=np.zeros([self.num_entity_class]).astype(np.float32),
                regularizer=self.regularizer,
            )
            hr_out = tf.nn.xw_plus_b(head_repre_b, w_e, b_e)
            tr_out = tf.nn.xw_plus_b(tail_repre_b, w_e, b_e)

        # de_out_dim = rnn_output_dim

        # 包的表示

        # with tf.variable_scope('sentence_attention') as scope:
        #     sentence_query = tf.get_variable('sentence_query', [de_out_dim, 1],
        #                                      initializer=tf.contrib.layers.xavier_initializer())

        #     def getSentenceAtt(num):
        #         num_sents = num[1] - num[0]
        #         bag_sents = sent_repre[num[0]:num[1]]

        #         sentence_att_weights = tf.nn.softmax(
        #             tf.reshape(tf.matmul(tf.tanh(bag_sents), sentence_query), [num_sents]))

        #         bag_repre_ = tf.reshape(
        #             tf.matmul(
        #                 tf.reshape(sentence_att_weights, [1, num_sents]),
        #                 bag_sents
        #             ), [de_out_dim]
        #         )
        #         return bag_repre_

        #     bag_repre = tf.map_fn(getSentenceAtt, sent_num, dtype=tf.float32)

        # with tf.variable_scope('fully_connected_layer') as scope:
        #     w = tf.get_variable('w', [de_out_dim, self.num_class], initializer=tf.contrib.layers.xavier_initializer(),
        #                         regularizer=self.regularizer)
        #     b = tf.get_variable('b', initializer=np.zeros([self.num_class]).astype(np.float32),
        #                         regularizer=self.regularizer)
        #     re_out = tf.nn.xw_plus_b(bag_repre, w, b)
        #     re_out=tf.nn.dropout(re_out,dropout)

        ner_logits = tf.nn.softmax(hr_out)
        ner_pred = tf.argmax(ner_logits, axis=1)
        ner_actual = tf.argmax(head_label, axis=1)
        ner_accuracy_ = tf.cast(
            tf.equal(ner_pred, ner_actual), tf.float32, name="ner_accu"
        )
        ner_accuracy = tf.reduce_mean(ner_accuracy_)

        head_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=hr_out, labels=head_label)
        )  # use sigmoid loss multi-label classification
        # tf.losses.add_loss(0.25*head_loss)
        tail_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=tr_out, labels=tail_label)
        )
        # tf.losses.add_loss(0.25*tail_loss)
        # re_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=re_out, labels=input_y))
        # tf.losses.add_loss(re_loss)
        # c = tf.get_variable('c',initializer=0.5,trainable=True)
        loss = head_loss + tail_loss
        if self.regularizer != None:
            loss += tf.contrib.layers.apply_regularization(
                self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            )
        # loss=tf.losses.get_total_loss(add_regularization_losses=False,name='total_loss')
        # summary.add_moving_summary(loss)
        loss = tf.identity(loss, name="total_loss")
        summary.add_moving_summary(loss, ner_accuracy)
        return loss

    def optimizer(self):
        lr = tf.get_variable("learning_rate", initializer=0.001, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        return optimizer.apply_grad_processors(
            opt, [GlobalNormClip(5), SummaryGradient()]
        )


class Model(ModelDesc):
    def __init__(self, params):
        self.params = params

        if self.params.l2 == 0.0:
            self.regularizer = None
        else:
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.params.l2)

        self.load_data()

    def load_data(self):
        data = pickle.load(open(self.params.dataset, "rb"))

        self.voc2id = data["voc2id"]
        self.id2voc = data["id2voc"]
        self.max_pos = data["max_pos"]
        self.num_class = len(data["rel2id"])
        self.num_entity_class = len(data["e_type2id"])
        self.num_deLabel = 1

        # get word list
        self.word_list = list(self.voc2id.items())
        self.word_list.sort(key=lambda x: x[1])
        self.word_list, _ = zip(*self.word_list)

    def inputs(self):
        return [
            tf.TensorSpec([None, None], tf.int32, "input_x"),  # Xs
            tf.TensorSpec([None, None], tf.int32, "input_pos1"),  # Pos1s
            tf.TensorSpec([None, None], tf.int32, "input_pos2"),  # Pos2s
            tf.TensorSpec([None], tf.int32, "head_pos"),  # HeadPoss
            tf.TensorSpec([None], tf.int32, "tail_pos"),  # TailPoss
            tf.TensorSpec([None], tf.int32, "x_len"),  # X_len
            tf.TensorSpec((), tf.int32, "seq_len"),  # max_seq_len
            tf.TensorSpec((), tf.int32, "total_sents"),  # total_sents
            tf.TensorSpec((), tf.int32, "total_bags"),  # total_bags
            tf.TensorSpec([None, 3], tf.int32, "sent_num"),  # SentNum
            tf.TensorSpec([None, None], tf.int32, "input_y"),  # ReLabels
            tf.TensorSpec([None, None], tf.float32, "head_label"),  # HeadLabels
            tf.TensorSpec([None, None], tf.float32, "tail_label"),  # TailLabels
            tf.TensorSpec((), tf.float32, "rec_dropout"),
            tf.TensorSpec((), tf.float32, "dropout"),
        ]

    def build_graph(
        self,
        input_x,
        input_pos1,
        input_pos2,
        head_pos,
        tail_pos,
        x_len,
        seq_len,
        total_sents,
        total_bags,
        sent_num,
        input_y,
        head_label,
        tail_label,
        rec_dropout,
        dropout,
    ):
        with tf.variable_scope("word_embedding") as scope:
            model = gensim.models.KeyedVectors.load_word2vec_format(
                self.params.embed_loc, binary=False
            )
            embed_init = getEmbeddings(
                model, self.word_list, self.params.word_embed_dim
            )
            _word_embeddings = tf.get_variable(
                "embeddings",
                initializer=embed_init,
                trainable=True,
                regularizer=self.regularizer,
            )
            word_pad = tf.zeros(
                [1, self.params.word_embed_dim]
            )  # word embedding for 'UNK'
            word_embeddings = tf.concat([word_pad, _word_embeddings], axis=0)

            pos1_embeddings = tf.get_variable(
                "pos1_embeddings",
                [self.max_pos, self.params.pos_dim],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True,
                regularizer=self.regularizer,
            )
            pos2_embeddings = tf.get_variable(
                "pos2_embeddings",
                [self.max_pos, self.params.pos_dim],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True,
                regularizer=self.regularizer,
            )

            word_embeded = tf.nn.embedding_lookup(word_embeddings, input_x)
            pos1_embeded = tf.nn.embedding_lookup(pos1_embeddings, input_pos1)
            pos2_embeded = tf.nn.embedding_lookup(pos2_embeddings, input_pos2)
            embeds = tf.concat([word_embeded, pos1_embeded, pos2_embeded], axis=2)

        with tf.variable_scope("Bi_rnn") as scope:
            fw_cell = tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(self.params.rnn_dim, name="FW_GRU"),
                output_keep_prob=rec_dropout,
            )
            bk_cell = tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(self.params.rnn_dim, name="BW_GRU"),
                output_keep_prob=rec_dropout,
            )
            val, state = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bk_cell, embeds, sequence_length=x_len, dtype=tf.float32
            )

            hidden_states = tf.concat((val[0], val[1]), axis=2)
            rnn_output_dim = self.params.rnn_dim * 2

        # word attention
        with tf.variable_scope("word_attention") as scope:
            word_query = tf.get_variable(
                "word_query",
                [rnn_output_dim, 1],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            sent_repre = tf.reshape(
                tf.matmul(
                    tf.reshape(
                        tf.nn.softmax(
                            tf.reshape(
                                tf.matmul(
                                    tf.reshape(
                                        tf.tanh(hidden_states),
                                        [total_sents * seq_len, rnn_output_dim],
                                    ),
                                    word_query,
                                ),
                                [total_sents, seq_len],
                            )
                        ),
                        [total_sents, 1, seq_len],
                    ),
                    hidden_states,
                ),
                [total_sents, rnn_output_dim],
            )
        # add entity type classification
        with tf.variable_scope("entity_type_classification"):
            entity_query = tf.get_variable(
                "head_query",
                [rnn_output_dim, 1],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            # 以句子中的词index建立索引
            s_idx = tf.range(0, total_sents, 1, dtype=tf.int32)
            head_index = tf.concat(
                [
                    tf.reshape(s_idx, [total_sents, 1]),
                    tf.reshape(head_pos, [total_sents, 1]),
                ],
                axis=-1,
            )
            tail_index = tf.concat(
                [
                    tf.reshape(s_idx, [total_sents, 1]),
                    tf.reshape(tail_pos, [total_sents, 1]),
                ],
                axis=-1,
            )
            # add null word vector
            word_hidden_states = tf.concat(
                [tf.zeros([total_sents, 1, rnn_output_dim]), hidden_states], axis=1
            )
            # extract head/tail entity's hidden state. size (total_sents,hidden_dim)
            head_repre_s = tf.gather_nd(
                word_hidden_states, head_index, name="head_entity_h_in_sentence"
            )
            tail_repre_s = tf.gather_nd(
                word_hidden_states, tail_index, name="tail_entity_h_in_sentence"
            )

            # 计算一个包中head实体的多个向量的att和
            def getHeadRepre(num):
                num_sents = num[1] - num[0]
                bag_sents = head_repre_s[num[0] : num[1]]

                head_att_weights = tf.nn.softmax(
                    tf.reshape(tf.matmul(tf.tanh(bag_sents), entity_query), [num_sents])
                )

                head_repre_ = tf.reshape(
                    tf.matmul(tf.reshape(head_att_weights, [1, num_sents]), bag_sents),
                    [rnn_output_dim],
                )
                return head_repre_

            # 计算一个包中tail实体的多个向量的att和
            def getTailRepre(num):
                num_sents = num[1] - num[0]
                bag_sents = tail_repre_s[num[0] : num[1]]

                tail_att_weights = tf.nn.softmax(
                    tf.reshape(tf.matmul(tf.tanh(bag_sents), entity_query), [num_sents])
                )

                tail_repre_ = tf.reshape(
                    tf.matmul(tf.reshape(tail_att_weights, [1, num_sents]), bag_sents),
                    [rnn_output_dim],
                )
                return tail_repre_

            # 一个batch中实体的向量表示
            head_repre_b = tf.map_fn(
                getHeadRepre, sent_num, dtype=tf.float32
            )  # dimension(batchsize,rnn_output_dim)
            tail_repre_b = tf.map_fn(getTailRepre, sent_num, dtype=tf.float32)

        # with tf.variable_scope('entity_fully_connected_layer') as scope:
        #     w_e = tf.get_variable('w', [rnn_output_dim, self.num_entity_class],
        #                           initializer=tf.contrib.layers.xavier_initializer(),
        #                           regularizer=self.regularizer)
        #     b_e = tf.get_variable('b', initializer=np.zeros([self.num_entity_class]).astype(np.float32),
        #                           regularizer=self.regularizer)
        #     hr_out = tf.nn.xw_plus_b(head_repre_b, w_e, b_e)
        #     tr_out = tf.nn.xw_plus_b(tail_repre_b, w_e, b_e)

        de_out_dim = rnn_output_dim

        # 包的表示

        with tf.variable_scope("sentence_attention") as scope:
            sentence_query = tf.get_variable(
                "sentence_query",
                [de_out_dim, 1],
                initializer=tf.contrib.layers.xavier_initializer(),
            )

            def getSentenceAtt(num):
                num_sents = num[1] - num[0]
                bag_sents = sent_repre[num[0] : num[1]]

                sentence_att_weights = tf.nn.softmax(
                    tf.reshape(
                        tf.matmul(tf.tanh(bag_sents), sentence_query), [num_sents]
                    )
                )

                bag_repre_ = tf.reshape(
                    tf.matmul(
                        tf.reshape(sentence_att_weights, [1, num_sents]), bag_sents
                    ),
                    [de_out_dim],
                )
                return bag_repre_

            bag_repre = tf.map_fn(getSentenceAtt, sent_num, dtype=tf.float32)

        bag_repre = tf.concat([bag_repre, head_repre_b, tail_repre_b], axis=-1)
        de_out_dim = 3 * de_out_dim

        with tf.variable_scope("fully_connected_layer") as scope:
            w = tf.get_variable(
                "w",
                [de_out_dim, self.num_class],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=self.regularizer,
            )
            b = tf.get_variable(
                "b",
                initializer=np.zeros([self.num_class]).astype(np.float32),
                regularizer=self.regularizer,
            )
            re_out = tf.nn.xw_plus_b(bag_repre, w, b)
            re_out = tf.nn.dropout(re_out, dropout)

        logits = tf.nn.softmax(re_out, name="logits")
        y_pred = tf.argmax(logits, axis=1, name="pred_y")
        y_actual = tf.argmax(input_y, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_actual), tf.float32))

        # head_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hr_out,
        #                                                                    labels=head_label))  # use sigmoid loss multi-label classification
        # tf.losses.add_loss(0.25*head_loss)
        # tail_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tr_out, labels=tail_label))
        # tf.losses.add_loss(0.25*tail_loss)
        re_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=re_out, labels=input_y)
        )
        # tf.losses.add_loss(re_loss)
        # c = tf.get_variable('c',initializer=0.5,trainable=True)
        # loss = re_loss + c * (head_loss + tail_loss)
        loss = re_loss
        if self.regularizer != None:
            loss += tf.contrib.layers.apply_regularization(
                self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            )
        # loss=tf.losses.get_total_loss(add_regularization_losses=False,name='total_loss')
        # summary.add_moving_summary(loss)
        loss = tf.identity(loss, name="total_loss")
        summary.add_moving_summary(loss, accuracy)
        return loss

    def optimizer(self):
        lr = tf.get_variable("learning_rate", initializer=0.001, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        return optimizer.apply_grad_processors(
            opt, [GlobalNormClip(5), SummaryGradient()]
        )


def getdata(path, isTrain):
    ds = LMDBSerializer.load(path, shuffle=isTrain)
    ds = getbatch(ds, 64, isTrain)
    if isTrain:
        ds = MultiProcessRunnerZMQ(ds, 4)
    return ds


def resume_train(ds_train, ds_test, params):
    return AutoResumeTrainConfig(
        always_resume=False,
        data=QueueInput(ds_train),
        session_init=get_model_loader("./train_log/er2/{}".format(params.model)),
        starting_epoch=params.start_epoch,
        callbacks=[
            ModelSaver(),
            StatMonitorParamSetter(
                "learning_rate", "total_loss", lambda x: x * 0.2, 0, 5
            ),
            HumanHyperParamSetter("learning_rate"),
            PeriodicTrigger(
                InferenceRunner(ds_test, [ScalarStats("total_loss")]), every_k_epochs=2
            ),
            MovingAverageSummary(),
            MergeAllSummaries(),
            GPUUtilizationTracker(),
            GPUMemoryTracker(),
        ],
        model=Model(params),
        max_epoch=params.start_epoch + 9,
    )


def get_config(ds_train, ds_test, params):
    return TrainConfig(
        data=QueueInput(ds_train),
        callbacks=[
            ModelSaver(),
            StatMonitorParamSetter(
                "learning_rate", "total_loss", lambda x: x * 0.2, 0, 5
            ),
            HumanHyperParamSetter("learning_rate"),
            PeriodicTrigger(
                InferenceRunner(
                    ds_test,
                    [
                        ScalarStats("total_loss"),
                        ClassificationError("ner_accu", "ner_accuracy"),
                    ],
                ),
                every_k_epochs=2,
            ),
            MovingAverageSummary(),
            MergeAllSummaries(),
            GPUUtilizationTracker(),
            GPUMemoryTracker(),
        ],
        model=WarmupModel(params),
        max_epoch=2,
    )


def predict(model, model_path, data_path):
    ds = getdata(data_path, False)
    pred_config = PredictConfig(
        model=model,
        session_init=get_model_loader(model_path),
        input_names=[
            "input_x",
            "input_pos1",
            "input_pos2",
            "head_pos",
            "tail_pos",
            "x_len",
            "seq_len",
            "total_sents",
            "total_bags",
            "sent_num",
            "input_y",
            "head_label",
            "tail_label",
            "rec_dropout",
            "dropout",
        ],
        output_names=["logits", "input_y"],
    )
    pred = SimpleDatasetPredictor(pred_config, ds)

    logit_list, label_list = [], []

    for output in pred.get_result():
        logit_list += output[0].tolist()
        label_list += output[1].tolist()

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data", dest="dataset", default="./params.pkl", help="params to use"
    )
    parser.add_argument("-gpu", dest="gpu", default="0", help="gpu to use")
    parser.add_argument(
        "-pos_dim",
        dest="pos_dim",
        default=10,
        type=int,
        help="dimension of positional embedding",
    )
    parser.add_argument(
        "-l2", dest="l2", default=0.001, type=float, help="l2 regularization"
    )
    parser.add_argument(
        "-embed_loc",
        dest="embed_loc",
        default="./glove/glove.6B.50d_word2vec.txt",
        help="embed location",
    )
    parser.add_argument(
        "-word_embed_dim",
        dest="word_embed_dim",
        default=50,
        type=int,
        help="word embed dimension",
    )
    parser.add_argument(
        "-restore",
        dest="restore",
        action="store_true",
        help="restore from the previous best model",
    )
    parser.add_argument(
        "-only_eval",
        dest="only_eval",
        action="store_true",
        help="Only evaluate pretrained model(skip training",
    )
    parser.add_argument(
        "-seed", dest="seed", default=1234, type=int, help="seed for randomization"
    )
    parser.add_argument(
        "-rnn_dim",
        dest="rnn_dim",
        default=192,
        type=int,
        help="hidden state dimension of Bi-RNN",
    )

    subparsers = parser.add_subparsers(title="command", dest="command")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument(
        "-name", dest="name", required=True, help="name of the run"
    )

    parser_predict = subparsers.add_parser("predict")
    parser_predict.add_argument("-model", dest="model", help="model for prediction")

    parser_resume = subparsers.add_parser("resume")
    parser_resume.add_argument(
        "-model", dest="model", required=True, help="name of the previous model"
    )
    parser_resume.add_argument(
        "-start_epoch",
        dest="start_epoch",
        type=int,
        help="number of the starting epoch",
    )

    args = parser.parse_args()
    set_gpu(args.gpu)
    if args.command == "train":

        # set seed
        tf.set_random_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        logger.auto_set_dir(action="k")

        ds = getdata("./mdb/train.mdb", True)

        dss = getdata("./mdb/test.mdb", False)
        config = get_config(ds, dss, args)
        launch_train_with_config(config, SimpleTrainer())

    elif args.command == "resume":
        logger.auto_set_dir(action="k")

        ds = getdata("./mdb/train.mdb", True)

        dss = getdata("./mdb/test.mdb", False)
        resume_config = resume_train(ds, dss, args)
        launch_train_with_config(resume_config, SimpleTrainer())

    elif args.command == "predict":
        with open("pn.txt", "w", encoding="utf-8") as f:

            for model in [str(13740 + i * 4580) for i in range(10)]:
                f.write(model + "\t")
                for pnpath in ["./mdb/pn1.mdb", "./mdb/pn2.mdb", "./mdb/pn3.mdb"]:
                    p100, p200, p300 = predict(
                        Model(args),
                        os.path.join("./train_log/er3/", "model-" + model),
                        pnpath,
                    )
                    logger.info(
                        "    {}:P@100:{}  P@200:{}  P@300:{}\n".format(
                            pnpath, p100, p200, p300
                        )
                    )
                    line = "{}\t{}\t{}\t".format(p100, p200, p300)
                    f.write(line)
                f.write("\n")
            f.close()
