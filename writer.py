#import tensorflow._api.v2.compat.v1 as tf
#tf.disable_v2_behavior()

from cmath import nan
import pickle as pkl
import random
from tqdm import tqdm
import numpy as np

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, MultiSearch
import elasticsearch.helpers
import time
import sys

import configparser
import logging
logging.basicConfig(level=logging.INFO)

#一些工具函数
def dump_pkl(filename, obj):
    with open(filename, 'wb') as f:
        pkl.dump(obj, f)
    print('{} dumped'.format(filename))

def dump_lines(filename, lines):
    with open(filename, 'w') as f:
        f.writelines(lines)
    print('{} writelines completed'.format(filename))

def sampling(ori_file, sampled_file, rate):
    sampled_lines = []
    with open(ori_file) as f:
        for line in tqdm(f):
            r = random.randint(1, int(1 / rate))
            if r == 1:
                sampled_lines.append(line)
    dump_lines(sampled_file, sampled_lines)

def select_pos_list(input_list, pos_list):
    return np.array(input_list)[pos_list].tolist()

def select_pos_str(input_str, pos_list):
    return ','.join(np.array(input_str.split(','))[pos_list].tolist())



#把retraval pool里的数据分成了10份，这是标号
cnt=10  #change

es = Elasticsearch('http://localhost:9200')
index_name="tma"+str(cnt)   #天猫数据集


es.indices.delete(index=index_name, ignore=[400, 404])
create_index_body = {
    "mappings": {
        "properties": {
        "tensor": {
            "type": "dense_vector",
            "dims": 80,
            "index": True,
            #"similarity": "dot_product"
            "similarity": "cosine"
            #"similarity": "l2_norm"
            }
        }
    },
}
es.indices.create(index=index_name, body=create_index_body, ignore=400)
print('index created')


with open("/NAS2020/Share/lining/rim_data/tmall/search_label.pkl", 'rb') as f:  #retraval pool里每个item的label
    labels=pkl.load(f)
t = time.time()

for i in range(153,172): #change
    line_no=400000*i
    #embedding后的retrival pool，除了最后一个文件，每个文件里数组维度都是400000*128
    path="/NAS2020/Share/lining/rim_data/tmall/emb_search/emb_search"+str(i)+".pkl" 
    with open(path, 'rb') as f:
        data = pkl.load(f)
    docs = []

    for j in range(len(data)):
        tensor=data[j][:80]     #change     #取前80维
        #line_item = line[:-1].split(',')
        #sync_id = line_item[self.sync_c_pos]
        label = labels[i*400000+j]
        
        doc = {
            'tensor':tensor,
            'label': label,
            'line_no': line_no
        }
        docs.append(doc)
        line_no += 1

    actions = [{
        '_op_type': 'index',
        '_index': index_name,  
        '_source': d
    } 
    for d in docs]
    print("start bulk::",i)
    print(len(actions))
    elasticsearch.helpers.bulk(es, actions,ignore=200,request_timeout=20)
    print("finish bulk::",i)
    print('data insert time: %.2f seconds' % (time.time() - t))
    print('last line_no is {}'.format(line_no))
    print(" ")
    log_path="log"+str(cnt)+".txt"
    with open(log_path, 'a') as f:
        results=[str(i),str(time.time()-t),str(line_no)]
        result_line = '\t'.join(results) + '\n'
        f.write(result_line)
