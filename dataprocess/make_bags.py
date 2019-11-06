import tensorflow as tf
import numpy as np
from utils import *
import unicodedata
import re
import collections

# 读取原始数据


def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalize_chars(w):
    if w == "-LRB-":
        return "("
    elif w == "-RRB-":
        return ")"
    elif w == "-LCB-":
        return "{"
    elif w == "-RCB-":
        return "}"
    elif w == "-LSB-":
        return "["
    elif w == "-RSB-":
        return "]"
    return w.replace(r"\/", "/").replace(r"\*", "*")


def normalize_word(w):
    return re.sub(r"\d", "0", normalize_chars(w).lower())


def clean_string(str):
    w = str
    w = unicode_to_ascii(w.strip())
    split = w.split()
    for i in range(len(split)):
        split[i] = normalize_word(split[i])
    w = " ".join(split)
    w = re.sub(r"-", " - ", w)
    w = re.sub(r"/", " / ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = w.rstrip().strip()
    return w


def readtraindata():
    with open("./data/riedel_train.json", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())

            _id = "{}_{}".format(data["sub"], data["obj"])
            train_data[_id]["head_id"] = data["sub_id"]
            train_data[_id]["tail_id"] = data["obj_id"]
            train_data[_id]["head"] = clean_string(data["sub"])
            train_data[_id]["tail"] = clean_string(data["obj"])

            train_data[_id]["rels"][
                relation2id.get(data["rel"], relation2id["NA"])
            ].append(
                {
                    "sent": clean_string(data["sent"]),
                    "sent_dep": data["openie"]["sentences"][0]["basicDependencies"],
                }
            )
            if i % 10000 == 0:
                print(
                    "reading raw train data completed {}/{},{}".format(
                        i,
                        miss_cnt,
                        time.strftime("%d_%m_%Y") + "_" + time.strftime("%H:%M:%S"),
                    )
                )


def readtestdata():
    with open("./data/riedel_test.json", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())

            _id = "{}_{}".format(data["sub"], data["obj"])
            test_data[_id]["head_id"] = data["sub_id"]
            test_data[_id]["tail_id"] = data["obj_id"]
            test_data[_id]["head"] = clean_string(data["sub"])
            test_data[_id]["tail"] = clean_string(data["obj"])
            test_data[_id]["rels"].add(relation2id.get(data["rel"], relation2id["NA"]))
            test_data[_id]["sents"].append(
                {
                    "sent": clean_string(data["sent"]),
                    "sent_dep": data["openie"]["sentences"][0]["basicDependencies"],
                }
            )
            if i % 10000 == 0:
                print(
                    "reading raw test data completed {}/{},{}".format(
                        i,
                        miss_cnt,
                        time.strftime("%d_%m_%Y") + "_" + time.strftime("%H:%M:%S"),
                    )
                )


# 将原始数据按包分组


def writetrainbags():
    with open("./data/train_bags.json", "w", encoding="utf-8") as f:
        for _id, data in train_data.items():
            for rel, sents in data["rels"].items():

                entry = {}
                entry["head"] = data["head"]
                entry["tail"] = data["tail"]
                entry["head_id"] = data["head_id"]
                entry["tail_id"] = data["tail_id"]
                entry["sentence"] = sents
                entry["relation"] = [rel]

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def writetestbags():
    with open("./data/test_bags.json", "w", encoding="utf-8") as f:
        for _id, data in test_data.items():

            entry = {}
            entry["head"] = data["head"]
            entry["tail"] = data["tail"]
            entry["head_id"] = data["head_id"]
            entry["tail_id"] = data["tail_id"]
            entry["sentence"] = data["sents"]
            entry["relation"] = list(data["rels"])

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


"""*******************************依存关系标签统计*****************************************"""


def dependency_label_statics():
    tag_counts = collections.Counter()
    train_tags = set()
    with open("./data/train_bags.json", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            for sent in data["sentence"]:
                for word in sent["sent_dep"]:
                    tag = word["dep"]
                    tag_counts[tag] += 1
                    train_tags.add(tag)
            if i % 10000 == 0:
                print("completed {}".format(i))
    with open("./data/test_bags.json", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            for sent in data["sentence"]:
                for word in sent["sent_dep"]:
                    tag = word["dep"]
                    tag_counts[tag] += 1
                    train_tags.add(tag)
            if i % 10000 == 0:
                print("completed {}".format(i))
    labels = sorted(tag_counts.keys())
    labels.remove("ROOT")
    labels.insert(0, "ROOT")
    labels.insert(40, "")  # 为缺失的label标签补位，在训练中mask掉
    label_mapping = {label: i for i, label in enumerate(labels)}
    print(label_mapping)
    with open("./data/dep2id.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f)


if __name__ == "__main__":
    relation2id = json.loads(open("./data/rel2id.json", encoding="utf-8").read())
    miss_cnt = 0
    count = 0
    train_data = ddict(lambda: {"rels": ddict(list)})
    test_data = ddict(lambda: {"sents": [], "rels": set()})
    readtraindata()
    readtestdata()
    writetrainbags()
    writetestbags()
    dependency_label_statics()
