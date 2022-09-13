import argparse

import numpy as np
import pandas as pd
import os


def deal_avazu(path):
    origin_train = pd.read_csv(os.path.join(path, 'train.csv'))
    print(origin_train.shape)
    label = origin_train[['click']].to_numpy().astype(np.int32)

    origin_train.drop(axis=1, columns=['id', 'click'], inplace=True)
    return origin_train, label


def deal_criteo(path):
    origin_train = pd.read_table(os.path.join(path, 'train.txt'), sep='\t', header=None)
    origin_train.fillna(-1, inplace=True)
    print(origin_train.shape)
    label = origin_train[[0]].to_numpy().astype(np.int32)
    origin_train.drop(axis=1, columns=[0], inplace=True)

    return origin_train, label


def common_deal(path, r, origin_train, label):
    train_num = origin_train.shape[0] // (r + 2)
    dtypes = dict(origin_train.dtypes)
    int_column, str_column = [], []
    for k, v in dtypes.items():
        if v == 'O':
            str_column.append(k)
        else:
            int_column.append(k)
    print(int_column, str_column)
    int_data = origin_train[int_column].to_numpy().astype(np.int32)
    str_data = origin_train[str_column].to_numpy().astype(str)
    del origin_train
    print(str_data)
    fields_nums = 0
    use_field = []
    for field in range(int_data.shape[1]):
        remap_index = np.unique(int_data[:, field], return_inverse=True)[1]
        if np.max(remap_index) < 1000000:
            int_data[:, field] = remap_index + fields_nums
            fields_nums += np.max(remap_index) + 1
            use_field.append(field)
    int_data = int_data[:, use_field]

    use_field = []
    for field in range(str_data.shape[1]):
        remap_index = np.unique(str_data[:, field], return_inverse=True)[1]
        if np.max(remap_index) < 1000000:
            str_data[:, field] = remap_index + fields_nums
            fields_nums += np.max(remap_index) + 1
            use_field.append(field)
    str_data = str_data[:, use_field]

    print(int_data, str_data)
    all_data = np.concatenate([int_data, str_data.astype(np.int32), label], axis=1)
    search_data, train_data, test_data = all_data[:train_num * r], \
                                         all_data[train_num * r:train_num * (r + 1)], all_data[-train_num:]

    save_path = os.path.join(path, '{}.csv')
    np.savetxt(save_path.format('search_pool'), search_data, fmt='%d', delimiter=',')
    np.savetxt(save_path.format('target_train'), train_data, fmt='%d', delimiter=',')
    np.savetxt(save_path.format('target_test'), test_data, fmt='%d', delimiter=',')


def deal(args):
    data_dir = os.path.join(args.dir, args.dataset)
    if args.dataset == 'avazu':
        origin_train, label = deal_avazu(data_dir)
    else:
        origin_train, label = deal_criteo(data_dir)
    common_deal(data_dir, args.ratio, origin_train, label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--dir', type=str, default='./data')
    # options
    parser.add_argument('-d', '--dataset', type=str, default='avazu', choices=['avazu', 'criteo'])
    parser.add_argument('-r', '--ratio', type=int, default=30)
    args = parser.parse_args()
    deal(args)
