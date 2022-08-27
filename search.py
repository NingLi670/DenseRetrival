import configparser
import pickle as pkl
import random
import sys

import numpy as np
from elasticsearch import Elasticsearch
from tqdm import tqdm


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


def dump_linesa(filename, lines):
    with open(filename, 'a') as f:
        f.writelines(lines)
    print('{} writelines completed'.format(filename))


class ESReader(object):
    def __init__(self,
                 index_name,
                 size,
                 host_url='http://localhost:9200'):

        self.es = Elasticsearch(host_url)
        self.index_name = index_name
        self.size = size

    # For RIM: not grouping by label, query_rim1 is for sequential data setting
    def query_rim1(self, queries):
        responses = []
        # ms = MultiSearch(using=self.es, index=self.index_name)
        for q in queries:
            # s = Search().query("match", line=q)[:self.size]
            # ms = ms.add(s)
            # responses = ms.execute()
            query = {
                "field": "tensor",
                "query_vector": q[:80],  # change
                "k": self.size,
                "num_candidates": 100  # change
            }
            response = self.es.knn_search(index=self.index_name, knn=query, request_timeout=120)  # change   #knn search
            responses.append(response)

        res_lineno_batch = []
        # res_line_batch = []
        label_batch = []
        for response in responses:
            # print("len of res:{}".format(len(response)))
            res_lineno = []
            # res_line = []
            labels = []
            for hit in response['hits']['hits']:
                res_lineno.append(str(hit['_source']['line_no']))
                # res_line.append(list(map(int, hit.line.split(','))))
                labels.append(str(hit['_source']['label']))
            if not res_lineno:
                res_lineno.append('-1')
                labels.append('0')
            res_lineno_batch.append(res_lineno)
            # res_line_batch.append(res_line)
            label_batch.append(labels)
        return res_lineno_batch, label_batch  # , res_line_batch


class queryGen(object):
    def __init__(self,
                 target_file,
                 batch_size):

        self.batch_size = batch_size

        with open(target_file, 'rb') as f:
            self.target_lines = pkl.load(f)
        self.dataset_size = len(self.target_lines)

        if self.dataset_size % self.batch_size == 0:
            self.total_step = int(self.dataset_size / self.batch_size)
        else:
            self.total_step = int(self.dataset_size / self.batch_size) + 1
        self.step = 0
        print('data loaded')

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.total_step:
            raise StopIteration

        # q_batch = []
        if self.step != self.total_step - 1:
            lines_batch = self.target_lines[self.step * self.batch_size: (self.step + 1) * self.batch_size]
        else:
            lines_batch = self.target_lines[self.step * self.batch_size:]

        # q_batch = [l[:64] for l in lines_batch]
        self.step += 1

        # return q_batch
        return lines_batch


def pre_search_rim(query_generator,
                   es_reader,
                   search_res_col_file,
                   search_res_label_file):
    search_res_col_lines = []
    search_res_label_lines = []
    cnt = 0
    for batch in tqdm(query_generator):
        # if cnt <650:
        #     cnt+=1
        #     continue

        res_lineno_batch, label_batch = es_reader.query_rim1(batch)

        search_res_col_lines += [(','.join(res) + '\n') for res in res_lineno_batch]
        search_res_label_lines += [(','.join(label) + '\n') for label in label_batch]
        print("finish batch", cnt)
        cnt = cnt + 1
        if cnt % 50 == 0:
            dump_linesa(search_res_col_file, search_res_col_lines)
            dump_linesa(search_res_label_file, search_res_label_lines)
            search_res_col_lines = []
            search_res_label_lines = []
            print("save::", cnt)

    dump_linesa(search_res_col_file, search_res_col_lines)
    dump_linesa(search_res_label_file, search_res_label_lines)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print('PLEASE INPUT [DATASET] [BATCH_SIZE] [RETRIEVE_SIZE] [MODE]')
        sys.exit(0)
    dataset = sys.argv[1]
    batch_size = int(sys.argv[2])
    size = int(sys.argv[3])
    mode = sys.argv[4]

    # read config file
    cnf = configparser.ConfigParser()
    cnf.read('config.ini')

    query_generator_train = queryGen("/NAS2020/Share/lining/rim_data/tmall/emb_train.pkl",
                                     # change #embedding后的训练集，维度2003514*128
                                     batch_size)
    print("get train query")

    query_generator_test = queryGen("/NAS2020/Share/lining/rim_data/tmall/emb_test.pkl",  # change
                                    batch_size)
    print("get test query")

    index = [10]  # 要检索的标号（1-10）

    for i in index:
        name = "tma" + str(i)
        es_reader = ESReader(name, size)

        print('target train pre searching...')
        pre_search_rim(query_generator_train,
                       es_reader,
                       "/NAS2020/Share/lining/rim_data/tmall/search_res_col_train_knn" + str(i) + ".txt",  # change
                       "/NAS2020/Share/lining/rim_data/tmall/search_res_label_train_knn" + str(i) + ".txt")  # change

        print('target test pre searching...')
        pre_search_rim(query_generator_test,
                       es_reader,
                       "/NAS2020/Share/lining/rim_data/tmall/search_res_col_test_knn" + str(i) + ".txt",  # change
                       "/NAS2020/Share/lining/rim_data/tmall/search_res_label_test_knn" + str(i) + ".txt")  # change
