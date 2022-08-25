writer.py : 把embedding后的retrival pool放入elasticsearch

search.py : 从elasticsearch里检索

RIM/code/rim_cp.py、RIM/code/eval_tmall.py : 获取embedding后的数据（已经完成，数据存放在/NAS2020/Share/lining/rim_data/tmall里）

newrim.py、newloader.py、newtrain.py ： 去掉embedding层的RIM模型，运行时需要修改config.ini中的一些文件位置
