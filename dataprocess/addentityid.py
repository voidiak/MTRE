from utils import *

entitytype = ddict(list)
# addset={'m.02ptk4j', 'm.054f3n', 'm.0fknfj', 'm.02wyjfb', 'm.02x4vlj', 'm.05b3gw', 'm.08g_35', 'm.0fy_8y', 'm.0czv43', 'm.0ftd0b', 'm.0bt1kt', 'm.04jq1fb', 'm.03gx8my', 'm.06xj82', 'm.02rh5dv', 'm.02649h'}
addset=pickle.load(open('empty_entity.pkl','rb'))
for word in ['addtrain','addtest']:
    with open('{}sent.out'.format(word),'r',encoding='utf8')as f:
        l1=[]
        l2=[]
        for line in f.readlines():
            line=line.strip()
            if not len(line):
                continue
            temp=line.split('\t')
            if len(temp)>1 and not temp[1].startswith('O'):#过滤掉‘O’
                l1.append(temp[0])
                l2.append(temp[1])
        i=0
        entitylist=[]
        while i<len(l2):
            if l2[i].startswith('B'):
                entityname=l1[i]
                j=1

                while i+j<len(l2) and l2[i+j].startswith('I'):
                    entityname+=('_'+l1[i+j])
                    j+=1


                entitylist.append(entityname)
                types=[]
                for item in l2[i].split('-')[1].split(','):
                    types.append(item)
                types=list(set(types))
                for item in types:
                    entitytype[entityname].append(item)
                i = i + j
            else:
                print('error')

with open('addentityname2type.json','w',encoding='utf8') as f:
    json.dump(entitytype,f)

id2type={}

for word in ['addtrain','addtest']:
    with open('{}id2name.json'.format(word),'r',encoding='utf8')as f:
        id2name=json.load(f)
        for id in id2name:
            id2type[id]=list(set(entitytype[id2name[id]]))

with open('addentityid2type.json','w',encoding='utf8')as f:
    json.dump(id2type,f,ensure_ascii=False)

with open('etype2id.json','r',encoding='utf-8')as f:
    etype2id=json.load(f)


with open('entity2id.json','r',encoding='utf-8') as f:
    entity2id=json.load(f)
with open('entity2id.json','w',encoding='utf-8') as f:
    for item in addset:
        types=id2type[item]
        entity2id[item]=[etype2id[type] for type in types]
    json.dump(entity2id,f,ensure_ascii=False)