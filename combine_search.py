import os

import numpy as np


def f():
    for mode in ['train', 'test']:
        base_plus = [0, 10000000, 20000000, 30000000, 40000000]
        data = [np.load('search_{}_{}.npz'.format(i, mode)) for i in range(len(base_plus))]
        data_index = [_['index'] for _ in data]
        data_score = [_['score'] for _ in data]
        for i in range(len(base_plus)):
            data_index[i] += base_plus[i]
        final_num = data_index[0].shape[1]
        data_index = np.concatenate(data_index, axis=-1)
        data_score = np.concatenate(data_score, axis=-1)

        top_index = np.argsort(-data_score, axis=-1)[:, :final_num]  # descending

        data_index, data_score = np.take_along_axis(data_index, top_index, -1), np.take_along_axis(data_score,
                                                                                                   top_index, -1)
        print(data_score)
        store_dir = {'index': data_index, 'score': data_score}
        np.savez(os.path.join('search_{}'.format(mode)), **store_dir)
        del data_index, data_score


f()
