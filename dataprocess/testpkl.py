from tensorpack import *
from tensorpack.dataflow import DataFlow, LMDBSerializer, DataFromList
import argparse
import pickle


class Raw(DataFlow):
    def __init__(self, data):
        self.data = data
        self.count = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for bag in data:
            # X = bag['X']
            # Pos1 = bag['Pos1']
            # Pos2 = bag['Pos2']
            # DepMask = bag['DepMask']
            HeadPos = bag["HeadPos"]
            TailPos = bag["TailPos"]
            if max(max(bag["HeadPos"]), max(bag["TailPos"])) > 100:
                self.count += 1
            # DepLabel = bag['Dep']
            # ReLabel = bag['Y']
            # HeadLabel = bag['HeadLabel']
            # TailLabel = bag['TailLabel']
            # output = [X, Pos1, Pos2, DepMask, HeadPos, TailPos, DepLabel, ReLabel, HeadLabel, TailLabel]
            output = [HeadPos, TailPos]
            yield output


if __name__ == "__main__":

    data = pickle.load(open("/data/PKL/train.pkl", "rb"))
    ds = Raw(data)
    LMDBSerializer.save(ds, "/data/MLRE/testpkl")
    print(ds.count)
