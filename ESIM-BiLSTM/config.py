# -*- encoding: utf-8 -*-
"""

@Project: ESIM
@File   : config.py
@Time   : "2019/7/21 8:52
@Author  : ZuoquanGong
@Email  : 2754915212@qq.com
@Software: PyCharm

"""
class Params:
    def __init__(self):
        # self.use_gpu=True
        self.use_gpu=False
        self.device_id=0
        self.device='cpu'

        self.train_path='data/SICK_train.txt'
        self.dev_path='data/SICK_trial.txt'
        self.test_path='data/SICK_test.txt'
        self.vocab_freq_cut=0#根据词频剪出词表
        self.max_sent_len=80

        # self.has_pretrain=True
        self.has_pretrain=False
        self.pretrain_path='pretrain/word2vec_40w_300.txt'

        self.drop_prob=0.5
        self.hidden_size=300
        self.output_size=3
        self.embed_dim=300

        self.epochs=20
        self.learning_rate=0.0003
        self.weight_decay=1e-7
        self.batch_size=32
        self.show_iter=20

        self.save_file='model_save/'
    def __show__(self):
        #__dict__函数可以获得一个dict，形式为{ 实例的属性 ：属性值 }
        print('\n[  Parameters  ]\n')
        for item in self.__dict__.items():
            print('{}: \'{}\''.format(item[0],item[1]))
        print()

