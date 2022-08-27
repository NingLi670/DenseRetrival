import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm


def deal1(data_dir):
    with open(os.path.join(data_dir, 'emb_train.pkl'), 'rb') as f:
        query_train = np.array(pickle.load(f)).astype('float32')
    print(query_train, query_train.shape)

    with open(os.path.join(data_dir, 'emb_test.pkl'), 'rb') as f:
        query_test = np.array(pickle.load(f)).astype('float32')
    print(query_test, query_test.shape)

    search_folder = os.path.join(data_dir, 'emb_search')
    search_pool = []
    for search_part in tqdm(os.listdir(search_folder)):
        with open(os.path.join(search_folder, search_part), 'rb') as f:
            search_pool.append(np.array(pickle.load(f)).astype('float32'))
    search_pool = np.concatenate(search_pool)
    print(search_pool, search_pool.shape)

    store_dir = {'train': query_train, 'test': query_test, 'search': search_pool}
    np.savez(os.path.join(data_dir, 'emb.npz'), **store_dir)


def deal(args):
    data_dir = os.path.join(args.data_dir, args.dataset)
    if args.dataset in ['tmall', 'taobao']:
        deal1(data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--data_dir', type=str, default='/NAS2020/Share/lining/rim_data/')
    # options
    parser.add_argument('-d', '--dataset', type=str, default='tmall', choices=['tmall', 'taobao'])

    args = parser.parse_args()
    deal(args)
