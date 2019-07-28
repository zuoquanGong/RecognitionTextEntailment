# -*- encoding: utf-8 -*-
"""

@Project: ESIM
@File   : Trainee.py
@Time   : "2019/7/21 8:11
@Author  : ZuoquanGong
@Email  : 2754915212@qq.com
@Software: PyCharm

"""

import torch.nn as  nn
import torch
from ESIM import Model
import torch.optim as optim

class Nebuch:
    def __init__(self,params):
        embed=params.embedding
        ext_embed=params.ext_embedding

        self.params=params
        self.model=Model(params,embed,ext_embed=ext_embed)
        if params.use_gpu:
            torch.backends.cudnn.enabled=True
            self.model=self.model.cuda()

        self.cross_entropy=nn.CrossEntropyLoss()
        if self.params.use_gpu:
            self.cross_entropy=self.cross_entropy.cuda()

        self.optimizer=optim.Adam(filter(lambda param: param.requires_grad,
                                         self.model.parameters()),
                                  lr=self.params.learning_rate,
                                  weight_decay=self.params.weight_decay)
        self.correct_rate=0.0
    def clear_grads(self):
        self.model.zero_grad()
    def forward_compute(self,batch_data):
        label_prob=self.model(batch_data)
        gold_label=batch_data.label.argmax(dim=1)
        self.loss=self.cross_entropy(label_prob,gold_label)
        self.loss.backward()
        predict_label=label_prob.argmax(dim=1)
        statistic_count=gold_label.eq(predict_label).sum()
        self.correct_rate+=statistic_count.item()/batch_data.size

    def opt_step(self):
        self.optimizer.step()
    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()
    def batch_show(self,idx):
        print('current:{:^10d}, cost:{:^10f}'.format(idx + 1, self.loss.data.item()))
    def correct_show(self,batch_num):
        print('correct rate:{:^10f}'.format(self.correct_rate/batch_num))
        self.correct_rate=0.0
