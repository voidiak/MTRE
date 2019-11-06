from utils import *

with open("addempty_train_sent.json", "r", encoding="utf-8") as f:
    seg_label = []
    for k, line in enumerate(f):
        data = json.loads(line.strip())
        e1_index = data["head_pos_list"]
        e2_index = data["tail_pos_list"]
        sentences = data["words"]
        for i, sentence in enumerate(sentences):
            seglist = []

            for j in range(len(sentence)):
                if j == e1_index[i]:
                    words = sentence[j].split("_")
                    seglist.append(words[0] + "\t" + "B-E")
                    for other in words[1:]:
                        seglist.append(other + "\t" + "I-E")
                elif j == e2_index[i]:
                    words = sentence[j].split("_")
                    seglist.append(words[0] + "\t" + "B-E")
                    for other in words[1:]:
                        seglist.append(other + "\t" + "I-E")
                else:
                    seglist.append(sentence[j] + "\t" + "O")

            for word in seglist:
                seg_label.append(word + "\n")
            seg_label.append("\n")


with open("addtrain.seg", "w", encoding="utf-8") as f:
    for word in seg_label:
        f.write(word)

with open("addempty_test_sent.json", "r", encoding="utf-8") as f:
    seg_label = []
    for k, line in enumerate(f):
        data = json.loads(line.strip())
        e1_index = data["head_pos_list"]
        e2_index = data["tail_pos_list"]
        sentences = data["words"]
        for i, sentence in enumerate(sentences):
            seglist = []

            for j in range(len(sentence)):
                if j == e1_index[i]:
                    words = sentence[j].split("_")
                    seglist.append(words[0] + "\t" + "B-E")
                    for other in words[1:]:
                        seglist.append(other + "\t" + "I-E")
                elif j == e2_index[i]:
                    words = sentence[j].split("_")
                    seglist.append(words[0] + "\t" + "B-E")
                    for other in words[1:]:
                        seglist.append(other + "\t" + "I-E")
                else:
                    seglist.append(sentence[j] + "\t" + "O")

            for word in seglist:
                seg_label.append(word + "\n")
            seg_label.append("\n")

with open("addtest.seg", "w", encoding="utf-8") as f:
    for word in seg_label:
        f.write(word)
