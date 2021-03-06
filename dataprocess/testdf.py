from tensorpack import *
import numpy as np
from utils import *


class getbatch(ProxyDataFlow):
    def __init__(self, ds, batch, isTrain):
        self.batch = batch
        self.ds = ds
        self.isTrain = isTrain

    def reset_state(self):
        self.ds.reset_state()

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

            Xs, X_len, Pos1s, Pos2s, DepMasks, DepLabels, max_seq_len = self.pad_dynamic(
                Xs, Pos1s, Pos2s, DepMasks, DepLabels
            )
            ReLabels = self.getOneHot(Y, 53)
            total_sents = num
            total_bags = len(Y)
            if not self.isTrain:
                dropout = 1.0
                rec_dropout = 1.0
            else:
                dropout = 0.8
                rec_dropout = 0.8
            # yield [Xs, Pos1s, Pos2s, HeadPoss, TailPoss, DepMasks, X_len, max_seq_len, total_sents, total_bags, SentNum,
            #        ReLabels, DepLabels, HeadLabels, TailLabels, rec_dropout, dropout]
            max_pos = max(max(HeadPoss), max(TailPoss))
            yield max_pos

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
            temp[i, : len(ele)] = ele[:seq_len]

        return temp


if __name__ == "__main__":
    path1 = "/data/MTRE/mdb/train.mdb"
    path2 = "/data/mdb/test.mdb"
    ds = LMDBSerializer.load(path1, shuffle=True)
    dse = getbatch(ds, 200, True)
    dse.reset_state()
    a = dse.__iter__()
    count = 0
    for b in a:
        # print('HeadPoss:{}\tTailPoss:{}'.format(b[0], b[1]))
        if b > 100:
            count += 1
    print(count)
    # print('Xs:{}\n Pos1s:{}\n Pos2s:{}\n HeadPoss:{}\n TailPoss:{}\n DepMasks:{}\n X_len:{}\n max_seq_len:{}\n '
    #       'total_sents:{}\n total_bags:{}\n SentNum:{}\n ReLabels:{}\n DepLabels:{}\n HeadLabels:{}\n TailLabels:{}\n'.
    #       format(b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13],b[14]))
