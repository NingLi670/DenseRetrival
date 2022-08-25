import sys
import os
import numpy as np
from dataloader import DataloaderRIM
from sklearn.metrics import *
import random
import time
import pickle as pkl
import math
from rim_cp import *
import logging
logging.basicConfig(level=logging.INFO)
import configparser
from utils import *

random.seed(1111)

SAVE_PATH_PREFIX = '../models/'
LOG_PATH_PREFIX = '../logs/'

DECAY_FACTOR = 1

def get_elapsed(start_time):
    return time.time() - start_time

def embed(
          feature_size, 
          eb_dim,
          s_num, 
          c_num, 
          label_num,
          dataloader,
          l2_norm
          ):
    tf.reset_default_graph()
    #tf.compat.v1.reset_default_graph()
    model = RIM(feature_size, eb_dim, s_num, c_num, label_num)
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model_dir = os.path.join(SAVE_PATH_PREFIX, "tmall", "10", "RIM_100_0.001_0.0001")
        if not os.path.exists(model_dir):
            logging.info("ERROR PATH!")
            exit(1)
        model_path = os.path.join(model_dir, 'ckpt')
        logging.info('start restore')
        model.restore(sess, model_path)
        logging.info('finish restore')

        #emb_target=np.ones([1,eb_dim*c_num])
        emb_target=[]
        cnt=0
        for batch_data in dataloader:
            #print("batch_data[3] type::",type(batch_data[3]),np.array(batch_data[3]).shape)

            #emb_target=model.embed(sess,batch_data,l2_norm)
            #emb_target=np.concatenate((emb_target,target))

            #print(type(emb_target),type(emb_target[0]),len(emb_target),len(emb_target[0]))
            #print(np.array(emb_target).shape)
            #print(emb_target[0])

            emb_target+=(model.embed(sess,batch_data,l2_norm).tolist())
            cnt+=1
            if cnt % 100 == 0:
                print("number",cnt,len(emb_target))
        print("in function len::",len(emb_target))
        return emb_target


if __name__ == '__main__':
    if len(sys.argv) < 4:
        logging.info("PLEASE INPUT [MODEL TYPE] [DATASET] [GPU]")
        sys.exit(0)
    model_type = sys.argv[1]
    dataset = sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]

    t=time.time()
    # read config file
    cnf_dataset = configparser.ConfigParser()
    cnf_dataset.read('../configs/config.ini')

    cnf_train = configparser.ConfigParser()
    cnf_train.read('../configs/train_params.ini')
    
    # get training params
    logging.info("get training params")
    batch_sizes = list(map(int, cnf_train.get(dataset, 'batch_sizes').split(',')))
    lrs = list(map(float, cnf_train.get(dataset, 'lrs').split(',')))
    l2_norms = list(map(float, cnf_train.get(dataset, 'l2_norms').split(',')))
    eb_dim = cnf_train.getint(dataset, 'eb_dim')
    eval_batch_size = cnf_train.getint(dataset, 'eval_batch_size')

    log_freq_protion = cnf_train.getint(dataset, 'log_freq_protion')
    eval_freq_protion = cnf_train.getint(dataset, 'eval_freq_protion')
    max_epoch = cnf_train.getint(dataset, 'max_epoch')

    # get dataset stats
    logging.info("get dataset stats")
    feature_size = cnf_dataset.getint(dataset, 'feat_size')
    s_num = cnf_dataset.getint(dataset, 's_num')
    c_num = cnf_dataset.getint(dataset, 'c_num') 
    label_num = cnf_dataset.getint(dataset, 'label_num')
    dataset_size = cnf_dataset.getint(dataset, 'dataset_size')
    shuffle = cnf_dataset.getboolean(dataset, 'shuffle')

    
########################获取test/train set的embedding特征#####################################
    # #dataloader
    # dataloader = DataloaderRIM(eval_batch_size,
    #                             cnf_dataset.get(dataset, 'remap_c_pos_list'),
    #                             s_num,
    #                             cnf_dataset.get(dataset, 'target_test_file'), #change
    #                             cnf_dataset.get(dataset, 'search_res_col_test_file'), #change
    #                             cnf_dataset.get(dataset, 'search_res_label_test_file'), #change
    #                             cnf_dataset.get(dataset, 'search_pool_file'),
    #                             False)
    
    # logging.info('got dataloader')

    # logging.info("call function")
    # target=embed(
    #       feature_size, 
    #       eb_dim,
    #       s_num, 
    #       c_num, 
    #       label_num,
    #       dataloader,
    #       1e-4
    #       )
    # logging.info("finish function")
    # print("search_res type::",type(target),type(target[0]),len(target))
    # logging.info("start save")
    # dump_pkl("/NAS2020/Share/lining/rim_data/tmall/emb_test.pkl",target)  #change

    # print("total time:",time.time()-t)
#########################################################################################
    

  

########################获取search pool的embedding特征#####################################
    for i in range(0,172):
        path="/NAS2020/Share/lining/rim_data/tmall/search/search"+str(i)+".csv"
        dataloader = DataloaderRIM(eval_batch_size,
                                cnf_dataset.get(dataset, 'remap_c_pos_list'),
                                s_num,
                                path,
                                cnf_dataset.get(dataset, 'search_res_col_test_file'), #无关紧要，不需要修改
                                cnf_dataset.get(dataset, 'search_res_label_test_file'), #无关紧要，不需要修改
                                cnf_dataset.get(dataset, 'search_pool_file'), 
                                False)
        logging.info('got dataloader')

        logging.info("call function")
        target=embed(
            feature_size, 
            eb_dim,
            s_num, 
            c_num, 
            label_num,
            dataloader,
            1e-4
            )
        logging.info("finish function")
        print("search_res type::",type(target),type(target[0]),len(target))
        logging.info("start save")
        filename="/NAS2020/Share/lining/rim_data/tmall/emb_search/emb_search"+str(i)+".pkl"
        dump_pkl(filename,target)
#########################################################################################
