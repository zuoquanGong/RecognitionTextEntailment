# -*- encoding: utf-8 -*-
"""

@Project: ESIM
@File   : template.py
@Time   : "2019/7/20 14:36
@Author  : ZuoquanGong
@Email  : 2754915212@qq.com
@Software: PyCharm

"""
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch import LongTensor,ByteTensor,FloatTensor
from typing import List,Dict
import torch
import random
# 数据模板
# 包括实例类（Instance），实例的list类（InstanceList），词表类（Vocabulary）
# 实例列表类中包括数据集加载、词表建立、索引映射、词向量加载和建立、batch建立 等功能

# Vocabulary: 词表
class Vocabulary(Dict):
    def __init__(self,
                 is_label=False):
        super(Vocabulary,self).__init__()
        self.is_label=is_label
        if not self.is_label:
            self.padding='<padding>'
            self.unknown='<unknown>'
            self.pad_id=0
            self.unknown_id=1
            self.id2item=[self.padding,self.unknown]
        else:
            self.id2item=[]

    # 字典创建
    def build(self,
              text_list: List[List[List[str]]]):
        for tlist in text_list:
            for sent in tlist:
                for word  in sent:
                    if word in self.keys():
                        self[word]+=1
                    else:
                        self[word]=1
        id2freq=sorted(self.items(),key=lambda item:item[1],reverse=True) # 依据词频进行排序
        self.id2item.extend([item[0] for item in id2freq])
        self.update({item:i for i,item in enumerate(self.id2item)})

    def __getitem__(self,
                    item: str or int):
        if isinstance(item,str):
            if item in self.keys():
                return self.get(item) ##
            else:
                return self.unknown_id
        if isinstance(item,int):
            if item<len(self.id2item):
                return self.id2item[item]
            else:
                exit('Error: data set->vocab->id2item: index out of range.')


# **Instance: 实例单元
class Instance:
    def __init__(self):
        self.sent_p=[]
        self.sent_h=[]
        self.label=''
        self.type=str

        #pretrain mapping
        self.sent_p_ext=''
        self.sent_h_ext=''

    def fill_a_sent(self,
                    line: str):
        sent=[]
        for word in line.split(' '):
            if word == '':
                continue
            if word[-1] == ',' or word[-1] == '.':
                sent.append(word[:-1])
                sent.append(word[-1])
            elif len(word)>1:
                if word[-2:] == "'s":
                    sent.append(word[:-2])
                    sent.append(word[-2:])
                else:
                    sent.append(word)
            else:
                sent.append(word)
        return sent

    def read(self,
             line: List[str]):
        self.sent_p=self.fill_a_sent(line[1])
        self.sent_h=self.fill_a_sent(line[2])
        self.label=line[4]


    def __str__(self)->str:  # 内置函数重写，格式化打印内部元素
        str_sent_p=''
        str_sent_h=''
        str_label=''
        if self.type==str:
            str_sent_p=' '.join(self.sent_p)
            str_sent_h=' '.join(self.sent_h)
            str_label=self.label
        elif self.type==int:
            str_sent_p=self.sent_p.__str__()
            str_sent_h=self.sent_h.__str__()
            str_label=str(self.label)

        if self.sent_p_ext!='' and self.sent_h_ext!='':
            str_sent_p_ext="\nsent_premise_ext: "+self.sent_p_ext.__str__()
            str_sent_h_ext="\nsent_hypothesis_ext: "+self.sent_h_ext.__str__()
        else:
            str_sent_p_ext=''
            str_sent_h_ext=''
        string=''
        string+="sent_premise: "
        string+=str_sent_p
        string+="\nsent_hypothesis: "
        string+=str_sent_h
        string+=str_sent_p_ext
        string+=str_sent_h_ext
        string+="\nlabel: "
        string+=str_label
        return string

# **InstanceList: 继承list类，装载Instance
# 功能：读写，建立词表，转换index，建立batch，加载预训练词向量
# 成员：Instance组成的list，词表Vocabulary，batch生成器，预训练词嵌入
class InstanceList(List):
    def __init__(self):
        super(InstanceList,self).__init__()
        self.sent_vocab=Vocabulary()
        self.label_vocab=Vocabulary(is_label=True)
        self.label_size=0
        self.has_pretrain=False
        self.embed_dim=0
        self.batch_size=0
        self.batch_num=0

    # **文件读取
    def reader(self,
                 path: str):
        with open(path,'r',encoding='utf-8') as fin:
            for i,line in  enumerate(fin.readlines()):
                line=line.strip()
                if line.strip() == '' or i == 0:  # 跳过空行和第一行，SICK里第一行为数据说明
                    continue
                inst=Instance()
                line=line.split('\t')
                inst.read(line)
                self.append(inst)

    # **创建词表
    def vocab_maker(self):
        all_sent_p=[inst.sent_p for inst in self]
        all_sent_h=[inst.sent_h for inst in self]
        all_label=[inst.label for inst in self]
        self.sent_vocab.build([all_sent_p,all_sent_h])
        self.label_vocab.build([[all_label]])
        self.label_size=len(self.label_vocab)

    # **文本映射成索引
    def txt2idx(self,ext_sent_vocab='',ext_label_vocab=''):
        if ext_sent_vocab!='':
            self.sent_vocab=ext_sent_vocab
        if ext_label_vocab!='':
            self.label_vocab=ext_label_vocab
            self.label_size=len(self.label_vocab)
        for inst in self:
            inst.sent_p=[self.sent_vocab[word] for word in inst.sent_p]
            inst.sent_h=[self.sent_vocab[word] for word in inst.sent_h]
            inst.label=self.label_vocab[inst.label]
            inst.type=int

    # **文本映射成索引，用于使用预训练词向量的时候
    def txt2idx_ext(self,ext_sent_vocab=''):
        self.has_pretrain=True
        if ext_sent_vocab!='':
            self.ext_vocab=ext_sent_vocab
        for inst in self:
            inst.sent_p_ext=[self.ext_vocab[word] for word in inst.sent_p]
            inst.sent_h_ext=[self.ext_vocab[word] for word in inst.sent_h]

    # InstanceList洗牌
    def shuffle(self):
        np.random.shuffle(self)

    # **batch生成器
    # 输出的batch 已经完成embedding映射
    # 输出的batch 内部根据序列长度进行了排序（用于pack_paded_sequence）
    def batch_generator(self,
                        batch_size: int,
                        use_gpu: bool,
                        device='cpu'):
        self.batch_size=batch_size
        self.batch_num=len(self)//batch_size
        class BatchData:
            def __init__(self,
                         sent_p: Variable,
                         sent_h: Variable,
                         mask_p: Variable,
                         mask_h: Variable,
                         label: Variable,
                         batch_ext_sents = None
                         ):
                self.sent_p=sent_p
                self.sent_h=sent_h
                self.mask_p=mask_p
                self.mask_h=mask_h
                self.label=label
                assert mask_p.size(-1)==sent_p.size(-1)
                assert mask_h.size(-1)==sent_h.size(-1)
                if batch_ext_sents:
                    self.sent_p_ext,self.sent_h_ext=batch_ext_sents[0], batch_ext_sents[1]
                else:
                    self.sent_p_ext,self.sent_h_ext = None, None
            @property
            def size(self):
                return self.label.size(0)
        for batch in range(self.batch_num):
            batch_inst=self[batch*batch_size:(batch+1)*batch_size]
            # all_sent_p=[inst.sent_p for inst in batch_inst]
            # all_sent_h=[inst.sent_h for inst in batch_inst]

            sent_p_max_len=max([len(inst.sent_p) for inst in batch_inst])
            sent_h_max_len=max([len(inst.sent_h) for inst in batch_inst])

            batch_sent_p=Variable(LongTensor(batch_size,sent_p_max_len).zero_())
            batch_sent_h=Variable(LongTensor(batch_size,sent_h_max_len).zero_())
            mask_p=Variable(ByteTensor(batch_size,sent_p_max_len).zero_())
            mask_h=Variable(ByteTensor(batch_size,sent_h_max_len).zero_())
            batch_label=Variable(LongTensor(batch_size,self.label_size).zero_())

            for idx in range(batch_size):

                sent_p=batch_inst[idx].sent_p
                sent_h=batch_inst[idx].sent_h
                label=batch_inst[idx].label
                sent_p_len=len(sent_p)
                sent_h_len=len(sent_h)

                batch_sent_p[idx,:sent_p_len]=LongTensor(sent_p)
                batch_sent_h[idx,:sent_h_len]=LongTensor(sent_h)
                mask_p[idx,:sent_p_len].fill_(1)
                mask_h[idx,:sent_h_len].fill_(1)

                for lid in range(self.label_size):
                    if lid==label:
                        batch_label[idx][lid]=1
            # batch_sent_p=self.embedding(batch_sent_p)
            # batch_sent_h=self.embedding(batch_sent_h)
            # mask_p_max=mask_p.sum(dim=1).max()
            # mask_h_max=mask_h.sum(dim=1).max()
            if use_gpu:
                batch_sent_p.to(device=device)
                batch_sent_h.to(device=device)
                mask_p.to(device=device)
                mask_h.to(device=device)

            if self.has_pretrain:
                batch_sent_p_ext=Variable(LongTensor(batch_size,sent_p_max_len).zero_())
                batch_sent_h_ext=Variable(LongTensor(batch_size,sent_h_max_len).zero_())
                for idx in range(batch_size):
                    sent_p_ext=batch_inst[idx].sent_p_ext
                    sent_h_ext=batch_inst[idx].sent_h_ext
                    sent_p_len=len(sent_p_ext)
                    sent_h_len=len(sent_h_ext)
                    batch_sent_p_ext[idx,:sent_p_len]=LongTensor(sent_p_ext)
                    batch_sent_h_ext[idx,:sent_h_len]=LongTensor(sent_h_ext)
                if use_gpu:
                    batch_sent_p_ext.to(device=device)
                    batch_sent_h_ext.to(device=device)
                # batch_sent_p_ext=self.ext_embedding(batch_sent_p_ext)
                # batch_sent_h_ext=self.ext_embedding(batch_sent_h_ext)

                batch_data=BatchData(batch_sent_p,batch_sent_h,mask_p,mask_h,batch_label,
                                     batch_ext_sents=(batch_sent_p_ext, batch_sent_h_ext))
            else:
                batch_data=BatchData(batch_sent_p,batch_sent_h,mask_p,mask_h,batch_label)
            yield batch_data  # 生成批数据类实例，通过迭代调用

    # 字典加载
    # 注：这里更新了原始字典
    def load_pretrain(self,
             path: str):
        idx=0
        # self.clear()  ###
        self.ext_vocab=Vocabulary()
        embed_dim=-1
        word2vec=[]
        with open(path,'r',encoding='utf-8') as fin:
            for line in fin.readlines():
                line.strip()
                line=line.split()
                if len(line)<2:
                    continue
                # if idx>2000:  ####
                #     break
                word,vec=line[0],line[1:]
                if embed_dim==-1:  # 判断vec维数是否合理
                    embed_dim=len(vec)
                else:
                    assert embed_dim==len(vec),'词向量维度不一致'

                self.ext_vocab[word]=idx
                word2vec.append(np.array(vec))
                idx+=1

        oov_list=[word for word in self.sent_vocab.keys() if word not in self.ext_vocab]
        oov_count=len(oov_list)
        print('oov list: ',oov_list)
        print('oov count: ',oov_count)
        print('oov ratio: %.3f %%'%(100*oov_count/len(self.sent_vocab)))
        self.embed_dim=embed_dim
        pretrain_weight=np.zeros((len(self.ext_vocab),embed_dim))
        for i,vec in enumerate(word2vec):
            pretrain_weight[i]=vec

        self.ext_embedding=nn.Embedding(len(self.ext_vocab),embed_dim)
        self.ext_embedding.weight.data.copy_(FloatTensor(pretrain_weight))
        self.ext_embedding.weight.requires_grad=False
        del(word2vec)
        del(pretrain_weight)
        print('Extra_embedding is done.')

    def build_pretrain(self,
                       embed_dim: int):
        if not self.has_pretrain:
            self.embed_dim=embed_dim
        self.embedding=nn.Embedding(len(self.sent_vocab),self.embed_dim)
        self.embedding.weight.requires_grad=True


# unit test
if __name__=='__main__':
    dataset=InstanceList()
    dataset.reader('data/SICK_trial.txt')
    print(dataset[0])
    print(len(dataset))
    dataset.vocab_maker()
    print(len(dataset.sent_vocab))
    print(dataset.label_vocab)
    print(dataset[0])
    dataset.load_pretrain('pretrain/word2vec_40w_300.txt')
    dataset.txt2idx()
    print(dataset[0])
    # dataset.shuffle()
    # print(dataset[0])
    dataset.build_pretrain(300)
    for i,b in enumerate(dataset.batch_generator(32)):
        # dataset.embedding(b.sent_p.long())
        # dataset.ext_embedding(b.sent_h.long())
        if i==0:
            print(b.sent_p.size())
            break



