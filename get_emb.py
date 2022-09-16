import argparse
import os.path

import numpy as np


def load_emb(args):
    import tensorflow as tf
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(args.checkpoint, './ckpt.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
        emb = sess.run('embedding/emb_mtx:0')
        np.save(os.path.join(args.dir, args.dataset, 'emb'), emb)


def get_emb(args):
    data_dir = os.path.join(args.dir, args.dataset)
    d = {'search': 'search_pool', 'train': 'target_train', 'test': 'target_test'}
    meta_emb = np.load(os.path.join(data_dir, 'emb.npy'))
    for k, v in d.items():
        data = np.loadtxt(os.path.join(data_dir, v + '.csv'), delimiter=',', dtype=np.int32)[:, :-1]
        emb = meta_emb[data].reshape((data.shape[0], -1))
        np.save(os.path.join(data_dir, 'emb_{}'.format(k)), emb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--dir', type=str, default='./data')
    # options
    parser.add_argument('-d', '--dataset', type=str, default='avazu', choices=['avazu', 'criteo'])
    parser.add_argument('-c', '--checkpoint', type=str)
    args = parser.parse_args()

    get_emb(args)
