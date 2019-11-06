from tensorpack import *
from tensorpack.dataflow import DataFlow, LMDBSerializer, DataFromList
from utils import *
import json

try:
    import cPickle as pickle
except:
    import pickle as pickle


class Raw(DataFromList):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        # X,Pos1,Pos2,DepMask,HeadPos,TailPos=[],[],[],[],[],[]
        # DepLabel,ReLabel,HeadLabel,TailLabel=[],[],[],[]
        for bag in data:
            X = bag["X"]
            Pos1 = bag["Pos1"]
            Pos2 = bag["Pos2"]
            DepMask = bag["DepMask"]
            HeadPos = bag["HeadPos"]
            TailPos = bag["TailPos"]
            DepLabel = bag["Dep"]
            ReLabel = bag["Y"]

            HeadLabel = bag["HeadLabel"]
            TailLabel = bag["TailLabel"]
            output = [
                X,
                Pos1,
                Pos2,
                DepMask,
                HeadPos,
                TailPos,
                DepLabel,
                ReLabel,
                HeadLabel,
                TailLabel,
            ]
            yield output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="command", dest="command")
    parser_db = subparsers.add_parser("build", help="build train/test database")
    parser_db.add_argument("--dataset", help="path to train/test data", required=True)
    parser_db.add_argument("--db", help="output lmdb file", required=True)
    parser_eval = subparsers.add_parser("eval", help="bulid p@n eval database")
    parser_eval.add_argument("--dataset", help="path to eval data", required=True)
    parser_eval.add_argument("--db", help="output eval lmdb file", required=True)
    args = parser.parse_args()
    if args.command == "build":
        data = pickle.load(open(args.dataset, "rb"))
        ds = Raw(data)
        LMDBSerializer.save(ds, args.db)
    elif args.command == "eval":
        data = pickle.load(open(args.dataset, "rb"))
        ds = Raw(data)
        LMDBSerializer.save(ds, args.db)
