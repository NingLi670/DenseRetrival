import argparse
import os

import numpy as np
from elasticsearch import Elasticsearch, helpers


def handle_es(index, data):
    # 构造迭代器
    for i, value in enumerate(data):
        yield {'_index': index, 'id': i, 'fea': ' '.join(value.astype(str))}


def save_to_es(args):
    es = Elasticsearch(hosts=['http://localhost:9200/']).options(
        request_timeout=20,
        retry_on_timeout=True,
        ignore_status=[400, 404]
    )
    data_dir = os.path.join(args.dir, args.dataset, 'search_pool.csv')
    data = np.loadtxt(data_dir, delimiter=',', dtype=np.int32)[:, :-1]
    print(data.shape)
    index = args.dataset
    es.indices.delete(index=index)
    es.indices.create(index=index)
    print("Begin Insert!")
    helpers.bulk(client=es, actions=handle_es(index, data))
    print("Finish Insert!")
    es.close()


def search(args):
    es = Elasticsearch(hosts=['http://localhost:9200/']).options(
        request_timeout=20000,
        retry_on_timeout=True,
        ignore_status=[400, 404]
    )
    result = es.search(index=args.dataset)
    print(result)
    for target in ['test']:
        target_data = np.loadtxt(os.path.join(args.dir, args.dataset, 'target_{}.csv'.format(target)),
                                 delimiter=',', dtype=str)[:, :-1]
        target_data = np.array_split(target_data, 100)
        print(target_data[0].shape)
        indexes, scores = [], []
        for sub_data in target_data:
            queries = []
            for data in sub_data:
                query = [{'index': args.dataset},
                         {'size': 30, 'query': {
                             "match": {'fea': ' '.join(data)}}}]
                queries.extend(query)
            result = es.msearch(index=args.dataset, searches=queries)['responses']
            print(result)
            for rs in result:
                rs = rs['hits']['hits']
                rs = list(map(lambda x: (x['_source']['id'], x['_score']), rs))
                index, score = zip(*rs)
                indexes.append(index)
                scores.append(score)
        indexes = np.array(indexes, dtype=np.int32)
        scores = np.array(scores, dtype=np.float32)
        store_dir = {'index': indexes, 'score': scores}
        np.savez(os.path.join(args.dir, args.dataset, 'search_{}'.format(target)), **store_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--dir', type=str, default='./data')
    # options
    parser.add_argument('-d', '--dataset', type=str, default='avazu', choices=['avazu', 'criteo'])
    args = parser.parse_args()

    # save_to_es(args)
    # time.sleep(10)
    search(args)
