import os
from utils import *
import json
from collections import defaultdict as ddict
import re

rel2id = json.loads(open("/data/MLRE/data/rel2id.json", encoding="utf-8").read())
id2rel = dict([(v, k) for k, v in rel2id.items()])
data = {"train": [], "test": []}
dep2id = json.loads(open("/data/MLRE/data/dep2id.json", encoding="utf-8").read())
id2dep = dict([(v, k) for k, v in dep2id.items()])
type2id = json.loads(open("/data/MLRE/data/etype2id.json", encoding="utf-8").read())
entity2typeid = json.loads(
    open("/data/MLRE/data/entity2id.json", encoding="utf8").read()
)


def read_file(file_path):
    null_sentence = 0
    temp = []
    with open(file_path, encoding="utf-8") as f:
        for k, line in enumerate(f):
            bag = json.loads(line.strip())

            pos1_list = []
            pos2_list = []
            head_pos_list = []
            tail_pos_list = []
            wrds_list = []
            dep_mask_list = []
            dep_list = []

            for sent in bag["sentence"]:  # 分词
                # sent["nlp"]=ddict({"tokens":list})
                sent["nlp"] = {}
                sent["nlp"]["sentences"] = []
                tokenlist = []
                tokens = sent["sent"].split()
                for index, word in enumerate(tokens):
                    token = {}
                    token["index"] = index
                    token["originalText"] = word
                    token["characterOffsetBegin"] = len(
                        " ".join(sent["sent"].split()[0:index])
                    ) + (1 if index != 0 else 0)
                    token["characterOffsetEnd"] = (
                        len(" ".join(sent["sent"].split()[0:index]))
                        + len(word)
                        + (1 if index != 0 else 0)
                    )
                    tokenlist.append(token)
                sent["nlp"]["sentences"].append({"tokens": tokenlist})

            count = 0

            for sent in bag["sentence"]:

                # 输出head词和tail词在句子中的索引位置到(list)head_start_off和(list)tail_start_off，先找一个词的索引位置，再找另一个词，且另一个词的索引位置必须在第一个词
                # 的第一个字之前或最后一个字之后
                # 实体词由一个字或两个字组成，先分为len(head)>len(tail)和len(head)<=len(tail)
                if len(bag["head"]) > len(bag["tail"]):  # head词比tail词长的时候
                    head_idx = [
                        i
                        for i, e in enumerate(sent["sent"].split())
                        if e == bag["head"]
                    ]  # 不计词间空格时的head词的词列表索引位置（考虑多个head词）
                    head_start_off = [
                        len(" ".join(sent["sent"].split()[0:idx]))
                        + (1 if idx != 0 else 0)
                        for idx in head_idx
                    ]  # 计入词间空格时head词的句子列表索引位置（假设idx=0时，head_start_off=0;idx=1时，head_start_off=2)(对中文分词有利）
                    if head_start_off == []:  # 如果head是两个字的词时，用空格取代下划线后，利用正则表达式匹配
                        rehead = "(\W)" + bag["head"].replace("_", " ") + "(\W|$)"
                        head_start_off = [
                            m.start()
                            for m in re.finditer(
                                rehead, " " + sent["sent"].replace("_", " ")
                            )
                        ]
                    reserve_span = [
                        (start_off, start_off + len(bag["head"]))
                        for start_off in head_start_off
                    ]  # head词的span，(第一个字的索引位置，最后一个字的索引位置）

                    tail_idx = [
                        i
                        for i, e in enumerate(sent["sent"].split())
                        if e == bag["tail"]
                    ]
                    tail_start_off = [
                        len(" ".join(sent["sent"].split()[0:idx]))
                        + (1 if idx != 0 else 0)
                        for idx in tail_idx
                    ]
                    if tail_start_off == []:
                        retail = "(\W)" + bag["tail"].replace("_", " ") + "(\W|$)"
                        tail_start_off = [
                            m.start()
                            for m in re.finditer(
                                retail, " " + sent["sent"].replace("_", " ")
                            )
                        ]
                    tail_start_off = [
                        off
                        for off in tail_start_off
                        if all(
                            [off < span[0] or off > span[1] for span in reserve_span]
                        )
                    ]  # 筛选tail_start_off，tail词的句子列表索引位置,必须满足在head词的第一个字之前，或在最后一个字之后
                else:  # head词和tail词一样长，或head词短于tail词
                    tail_idx = [
                        i
                        for i, e in enumerate(sent["sent"].split())
                        if e == bag["tail"]
                    ]
                    tail_start_off = [
                        len(" ".join(sent["sent"].split()[0:idx]))
                        + (1 if idx != 0 else 0)
                        for idx in tail_idx
                    ]
                    if (
                        tail_start_off == []
                    ):  # 把句子中的下划线替换成空格后再查找实体位置，start()返回的是pattern开始的位置
                        retail = "(\W)" + bag["tail"].replace("_", " ") + "(\W|$)"
                        tail_start_off = [
                            m.start()
                            for m in re.finditer(
                                retail, " " + sent["sent"].replace("_", " ")
                            )
                        ]
                    reserve_span = [
                        (start_off, start_off + len(bag["tail"]))
                        for start_off in tail_start_off
                    ]  # tail词的span
                    head_idx = [
                        i
                        for i, e in enumerate(sent["sent"].split())
                        if e == bag["head"]
                    ]
                    head_start_off = [
                        len(" ".join(sent["sent"].split()[0:idx]))
                        + (1 if idx != 0 else 0)
                        for idx in head_idx
                    ]
                    if head_start_off == []:
                        rehead = "(\W)" + bag["head"].replace("_", " ") + "(\W|$)"
                        head_start_off = [
                            m.start()
                            for m in re.finditer(
                                rehead, " " + sent["sent"].replace("_", " ")
                            )
                        ]
                    head_start_off = [
                        off
                        for off in head_start_off
                        if all(
                            [off < span[0] or off > span[1] for span in reserve_span]
                        )
                    ]
                # '词span元组[(开始位置,结束位置,"词名"),...]')
                head_off = [
                    (head_off, head_off + len(bag["head"]), "head")
                    for head_off in head_start_off
                ]
                tail_off = [
                    (tail_off, tail_off + len(bag["tail"]), "tail")
                    for tail_off in tail_start_off
                ]
                if head_off == [] or tail_off == []:
                    # print('head tail off null')
                    # print('{} | {} | {}'.format(bag['head'], bag['tail'], sent['sent']))
                    # pdb.set_trace()
                    continue
                spans = [head_off[0]] + [tail_off[0]]
                off_begin, off_end, _ = zip(*spans)

                tid_map, tid2wrd = (
                    ddict(dict),
                    ddict(list),
                )  # tid_map:按空格分词的序列的下标到按实体分词的序列的下表的映射；tid2wrd按实体分词的序列

                tok_idx = 1
                head_pos, tail_pos = None, None
                for s_n, sentence in enumerate(sent["nlp"]["sentences"]):
                    i, tokens = 0, sentence["tokens"]
                    while i < len(tokens):
                        # print('sent order {}'.format(i))
                        if tokens[i]["characterOffsetBegin"] in off_begin:
                            _, end_offset, identity = spans[
                                off_begin.index(tokens[i]["characterOffsetBegin"])
                            ]

                            if identity == "head":
                                head_pos = tok_idx - 1  # Indexing starts from 0
                                # tok_list = [tok['originalText'] for tok in tokens]
                            else:
                                tail_pos = tok_idx - 1
                                # tok_list = [tok['originalText'] for tok in tokens]

                            while (
                                i < len(tokens)
                                and tokens[i]["characterOffsetEnd"] <= end_offset
                            ):
                                tid_map[s_n][tokens[i]["index"]] = tok_idx
                                tid2wrd[tok_idx].append(tokens[i]["originalText"])
                                i += 1

                            tok_idx += 1
                        else:
                            tid_map[s_n][tokens[i]["index"]] = tok_idx
                            tid2wrd[tok_idx].append(tokens[i]["originalText"])

                            i += 1
                            tok_idx += 1
                if head_pos == None or tail_pos == None:
                    # print('Skipped entry!!')
                    # print('{} | {} | {}'.format(bag['head'], bag['tail'], sent['sent']))
                    null_sentence += 1
                    # pdb.set_trace()
                    continue
                wrds = ["_".join(e).lower() for e in tid2wrd.values()]
                pos1 = [
                    i - head_pos for i in range(tok_idx - 1)
                ]  # tok_idx = (number of tokens + 1)
                pos2 = [i - tail_pos for i in range(tok_idx - 1)]

                def mappos(p):
                    if p > 99:
                        p = 0
                    else:
                        p = p + 1
                    return p

                # 当headpos超出max length的时候，置为0
                mapped_head_pos = mappos(head_pos)
                mapped_tail_pos = mappos(tail_pos)
                # entity_max_pos = max(mapped_head_pos, mapped_tail_pos)
                # if entity_max_pos>100:
                #     print('Skipped entry!!')
                #     print('{} | {} | {}'.format(bag['head'], bag['tail'], sent['sent']))
                #     pdb.set_trace()
                #     continue
                dep_mask = []  # 用于dependency parsing中标记词语位置的mask
                # 构建dependency label list
                dep_head_dict = ddict(int)
                dep_label_dict = ddict(str)
                # 首先建立按word序号索引的dependency head index dict
                word_id2dep_head = ddict(int)
                word_id2dep_label = ddict(str)
                for i, item in enumerate(sent["sent_dep"]):
                    word_id2dep_head[item["dependent"] - 1] = item["governor"]
                    word_id2dep_label[item["dependent"] - 1] = item["dep"]
                # 按word分词序列逐个修改dep_head
                for s_n, sentence in enumerate(sent["nlp"]["sentences"]):
                    i, j, tokens = 0, 0, sentence["tokens"]
                    tok_idx = 1
                    while i < len(tokens):
                        if tokens[i]["characterOffsetBegin"] in off_begin:
                            _, end_offset, identity = spans[
                                off_begin.index(tokens[i]["characterOffsetBegin"])
                            ]
                            while (
                                i < len(tokens)
                                and tokens[i]["characterOffsetEnd"] <= end_offset
                            ):
                                for key in word_id2dep_head:
                                    if (
                                        word_id2dep_head[key] == i + 1
                                    ):  # dependent id=idex+1
                                        word_id2dep_head[key] = tok_idx

                                i += 1
                            dep_label_dict[tok_idx] = word_id2dep_label[i - 1]
                            tok_idx += 1
                        else:
                            for key in word_id2dep_head:
                                if word_id2dep_head[key] == i + 1:
                                    word_id2dep_head[key] = tok_idx
                            dep_label_dict[tok_idx] = word_id2dep_label[i]

                            i += 1
                            tok_idx += 1
                    # 此时word_id2dep_head应该已经清洗过一遍
                    tok_idx = 1
                    while j < len(tokens):
                        if tokens[j]["characterOffsetBegin"] in off_begin:
                            _, end_offset, identity = spans[
                                off_begin.index(tokens[j]["characterOffsetBegin"])
                            ]
                            while (
                                j < len(tokens)
                                and tokens[j]["characterOffsetEnd"] <= end_offset
                            ):
                                j += 1
                            dep_head_dict[tok_idx] = word_id2dep_head[j - 1]
                            tok_idx += 1
                        else:
                            dep_head_dict[tok_idx] = word_id2dep_head[j]
                            j += 1
                            tok_idx += 1
                # 将dep_head_dict和dep_label_dict转成list
                dep_head = [item for item in dep_head_dict.values()]
                dep_label = [item for item in dep_label_dict.values()]
                sen_length = len(dep_head)
                mapped_pos = (
                    mapped_head_pos,
                    mapped_tail_pos,
                )  # 由于dep_head的索引是从1开始计数，所以正好和map后的entity pos一致
                len_limit = min(100, sen_length) + 1
                dep = []
                # masked dependency label
                # if len(dep_head) == len(dep_label):
                #     for i in range(sen_length):
                #         if dep_head[i] < len_limit and dep_label[i] != 40:
                #             if (dep_head[i] in mapped_pos) or (i in entity_pos):
                #                 dep.append(dep_head[i])
                #                 dep_mask.append(1)
                #                 #TO-DO:加入dependency arc label标签arc_rel
                #             else:
                #                 dep.append(0)
                #                 dep_mask.append(0)
                #         else:
                #             dep.append(0)
                #             dep_mask.append(0)
                for i in range(sen_length):
                    if dep_head[i] < len_limit and dep_label[i] != 40:
                        dep.append(dep_head[i])
                        dep_mask.append(1)
                    else:
                        dep.append(0)
                        dep_mask.append(0)
                dep_list.append(dep)
                wrds_list.append(wrds)
                pos1_list.append(pos1)
                pos2_list.append(pos2)
                head_pos_list.append(head_pos)
                tail_pos_list.append(tail_pos)
                if max(head_pos_list) > 100:
                    print(head_pos_list)
                    pdb.set_trace()
                dep_mask_list.append(dep_mask)
                count += 1

            head_label = [0] * len(type2id)
            tail_label = [0] * len(type2id)

            # standard operation
            for i in entity2typeid[bag["head_id"]]:
                head_label[i] = 1
            for i in entity2typeid[bag["tail_id"]]:
                tail_label[i] = 1

            entity_max_pos = max(max(head_pos_list), max(tail_pos_list))
            if entity_max_pos > 100:
                print("Skipped entry!!")
                print("{} | {} | {}".format(bag["head"], bag["tail"], sent["sent"]))
                pdb.set_trace()
                continue

            temp.append(
                {
                    "head": bag["head"],
                    "tail": bag["tail"],
                    "rels": bag["relation"],
                    "head_label": head_label,
                    "tail_label": tail_label,
                    "head_pos_list": head_pos_list,
                    "tail_pos_list": tail_pos_list,
                    "wrds_list": wrds_list,
                    "pos1_list": pos1_list,
                    "pos2_list": pos2_list,
                    "dep_mask_list": dep_mask_list,
                    "dep_list": dep_list,
                }
            )

    print("{} null sentences:{}".format(file_path, null_sentence))
    return temp


data["train"] = read_file("/data/MLRE/data/train_bags.json")
data["test"] = read_file("/data/MLRE/data/test_bags.json")

print("Bags processed:Train:{},Test:{}".format(len(data["train"]), len(data["test"])))
