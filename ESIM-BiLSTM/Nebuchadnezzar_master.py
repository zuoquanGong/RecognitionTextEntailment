# -*- encoding: utf-8 -*-
"""

@Project: ESIM
@File   : Nebuchadnezzar_master.py
@Time   : "2019/7/20 21:16
@Author  : ZuoquanGong
@Email  : 2754915212@qq.com
@Software: PyCharm

"""
from template import InstanceList
from Trainee import Nebuch

class Nebuchadnezzar:
    def __init__(self,
                 params):
        self.params = params
        self.train = InstanceList()
        self.dev = InstanceList()
        self.test = InstanceList()
        self.trainee = None
    # 1.数据预处理
    def data_process(self):
        self.train.reader(self.params.train_path)
        self.dev.reader(self.params.dev_path)
        self.test.reader(self.params.test_path)

        self.train.vocab_maker()
        if self.params.has_pretrain:
            self.train.load_pretrain(self.params.pretrain_path)
            self.train.txt2idx_ext()
            self.dev.txt2idx_ext(ext_sent_vocab=self.train.ext_vocab)
            self.test.txt2idx_ext(ext_sent_vocab=self.train.ext_vocab)
            # 这里我们维持了两套 word-embedding ，输入模型的是外部 embedding 和语料 embedding 的加和
            # 外部引入的不会随着模型训练而改变，语料embedding会随训练而改变

        self.train.txt2idx()
        self.dev.txt2idx(ext_sent_vocab=self.train.sent_vocab,
                        ext_label_vocab=self.train.label_vocab)
        self.test.txt2idx(ext_sent_vocab=self.train.sent_vocab,
                        ext_label_vocab=self.train.label_vocab)

        print(self.train[0])
        print(self.dev[0])
        print(self.test[0])

        self.train.build_pretrain(self.params.embed_dim)
        self.dev.build_pretrain(self.params.embed_dim)
        self.test.build_pretrain(self.params.embed_dim)
        print('train data set size: ', len(self.train))
        print('dev data set size: ', len(self.dev))
        print('test data set size: ', len(self.test))

    # 2.模型训练
    def train_process(self):
        # 以下两行为ESIM模型传递 embeding
        self.params.embedding = self.train.embedding
        if self.params.has_pretrain:
            self.params.ext_embedding = self.train.ext_embedding
        else:
            self.params.ext_embedding = None

        self.trainee = Nebuch(self.params)
        for epoch in range(self.params.epochs):
            print('\n[  Iteration %s  ]\n' % str(epoch+1))
            self.train.shuffle()
            self.trainee.train()
            for update_iter, batch_data in enumerate(
                    self.train.batch_generator(self.params.batch_size, self.params.use_gpu)):
                self.trainee.clear_grads()
                self.trainee.forward_compute(batch_data)
                self.trainee.opt_step()

                if update_iter % self.params.show_iter == 0 and update_iter != 0:
                    self.trainee.batch_show(update_iter)
            self.trainee.correct_show(self.train.batch_num)
            self.trainee.eval()
            print('dev data set: ')
            self.eval_process(self.dev)
            print('test data set: ')
            self.eval_process(self.test)

    # 3.模型测试
    def eval_process(self,
                     dataset: InstanceList):
        for update_iter, batch_data in enumerate(
                dataset.batch_generator(self.params.batch_size, self.params.use_gpu)):
            self.trainee.forward_compute(batch_data)
        self.trainee.correct_show(dataset.batch_num)
