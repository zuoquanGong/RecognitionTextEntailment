# -*- encoding: utf-8 -*-
"""

@Project: ESIM
@File   : main.py.py
@Time   : "2019/7/20 11:00
@Author  : ZuoquanGong
@Email  : 2754915212@qq.com
@Software: PyCharm

"""
from optparse import OptionParser
from Nebuchadnezzar_master import Nebuchadnezzar
from config import Params
import torch
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s - %(lkevelname)s - %(message)s',
                    level=logging.INFO)

if __name__ == '__main__':
    # 接受命令行输入
    # 接受config配置文件的配置参数
    # 初始化参数params
    # 调用master，进入模型训练模块

    #基础程序设置
    seed=789
    torch.manual_seed(seed)#seed: long
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    #设备信息
    print('GPU available: ',torch.cuda.is_available())
    print('CUDNN available: ',torch.backends.cudnn.enabled)
    print('GPU number: ',torch.cuda.device_count())

    # parser = OptionParser()
    # parser.add_option("--train", dest="trainFile",
    #                   default="", help="train dataset")
    # parser.add_option("--dev", dest="devFile",
    #                   default="", help="dev dataset")
    # parser.add_option("--test", dest="testFile",
    #                   default="", help="test dataset")

    # (options, args) = parser.parse_args()

    params=Params()
    if not torch.cuda.is_available() or not torch.backends.cudnn.enabled or torch.cuda.device_count()<=0:
        params.use_gpu=False
    params.device=torch.device("cuda:"+str(params.device_id) if params.use_gpu else 'cpu')

    master=Nebuchadnezzar(params)
    master.data_process()
    master.train_process()
