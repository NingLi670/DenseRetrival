import argparse
import os
import time

import faiss
import numpy as np


def search(args):
    data_dir = os.path.join(args.data_dir, args.dataset)
    t1 = time.perf_counter()
    emb_search = np.load(os.path.join(data_dir, 'emb_search.npy'))
    print("Data Loaded! Costed {}s".format(time.perf_counter() - t1))

    index = faiss.IndexFlatIP(args.dim)  # build the index
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(normalize(emb_search[40000000:, :args.dim]))
    print("Index Add!")
    for target in ['train', 'test']:
        t1 = time.perf_counter()
        emb = np.load(os.path.join(data_dir, 'emb_{}.npy'.format(target)))
        target_score, target_index = index.search(normalize(emb[:, :args.dim]), args.candidate)  # actual search
        target_index = target_index.astype('int32')
        store_dir = {'index': target_index, 'score': target_score}
        np.savez(os.path.join(data_dir, 'search_4_{}'.format(target)), **store_dir)
        print(time.perf_counter() - t1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--data_dir', type=str, default='/NAS2020/Share/lining/rim_data/')
    # options
    parser.add_argument('-d', '--dataset', type=str, default='tmall',
                        choices=['tmall', 'taobao', 'alipay', 'avazu', 'criteo'])
    parser.add_argument('--dim', type=int, default=64, help='Dimension used to calculate similarity')
    parser.add_argument('--candidate', type=int, default=100)
    normalize = lambda x: x / np.sqrt(np.sum(x ** 2, axis=-1, keepdims=True) + 1e-9)

    args = parser.parse_args()
    search(args)
