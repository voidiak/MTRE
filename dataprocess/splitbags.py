import pickle
def splitBags(data, chunk_size):
    delbag = 0
    addbag = 0
    for i in range(len(data) - 1, -1, -1):
        bag = data[i]
        if len(bag['X']) > chunk_size:
            delbag += 1
            del data[i]
            chunks = getChunks(range(len(bag['X'])), chunk_size)

            for chunk in chunks:
                res={}
                res['Y'] = bag['Y']
                res['HeadLabel'] = bag['HeadLabel']
                res['TailLabel'] = bag['TailLabel']
                res['X'] = [bag['X'][j] for j in chunk]
                res['Pos1'] = [bag['Pos1'][j] for j in chunk]
                res['Pos2'] = [bag['Pos2'][j] for j in chunk]
                res['HeadPos'] = [bag['HeadPos'][j] for j in chunk]
                res['TailPos'] = [bag['TailPos'][j] for j in chunk]
                res['DepMask'] = [bag['DepMask'][j] for j in chunk]
                res['Dep'] = [bag['Dep'][j] for j in chunk]

                data.append(res)
                addbag += 1

    print('deleted bag :{}  added bag :{}'.format(delbag, addbag))
    return data


if __name__ == '__main__':
    data=pickle.load(open('/data/MLRE-NG/PKL/pn1_r.pkl','rb'))
    print(len(data))
    data_splited=splitBags(data)