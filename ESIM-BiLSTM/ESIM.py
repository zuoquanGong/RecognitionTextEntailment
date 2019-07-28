# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:50:08 2019

@author: zuoquan gong
"""

import torch.nn as  nn
import torch
import torch.autograd
import torch.nn.functional as F
# =============================================================================
# ESIM: Enhanced Sequential Inference Model
# =============================================================================
class Model( nn.Module ):
    
    def __init__(self, hyper_params,embed,ext_embed=None):
        super(Model, self).__init__()
        self.info_name='ESIM'
        self.info_task='Natural Language Inference'
        self.info_func='all'

        self.embed_dim=hyper_params.embed_dim
        self.has_pretrain = hyper_params.has_pretrain
        self.embed = embed
        if self.has_pretrain:
            self.embed_ext = ext_embed
        print(self.embed)
        self.hidden_size=hyper_params.hidden_size
        self.linear_size=hyper_params.hidden_size
        self.label_size=hyper_params.output_size
        self.dropout_prob=hyper_params.drop_prob

        # 1. bilstm encoder
        self.lstm1 = nn.LSTM(self.embed_dim, self.hidden_size,  batch_first=True, bidirectional=True)
        # 2. softmax attention
        # self.soft_assign=self.softmax_attention()
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size*4*2,self.hidden_size),
            nn.ReLU()
        )
        # print('projection dim: ',self.hidden_size*8)
        # 3.composition
        self.lstm2=nn.LSTM(self.hidden_size,self.hidden_size,batch_first=True,bidirectional=True)
        # 4.classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(8 * self.hidden_size),
            nn.Linear(8 * self.hidden_size, self.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.linear_size),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.linear_size, self.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.linear_size),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.linear_size, 3),
            nn.Softmax(dim=-1)
        )
        self.apply(_init_esim_weights)  # 通过递归完成所有权重的初始化
        # for params in self.named_parameters():
        #     print(params)

    # 两句话（前提-premise、假设-hypothesis）之间做attention
    def softmax_attention(self,sent_p,sent_h, mask_p,mask_h):
        attention=sent_p.bmm(sent_h.transpose(1,2))

        # change '1.' in the mask tensor to '-inf'
        mask_p = 1 - mask_p
        mask_p = mask_p.float().masked_fill_(mask_p, float('-inf'))  # b,sp
        mask_h = 1 - mask_h
        mask_h = mask_h.float().masked_fill_(mask_h, float('-inf'))  # b,sh

        mask_h = mask_h.unsqueeze(1).expand_as(attention)
        mask_p = mask_p.unsqueeze(1).expand_as(attention.transpose(1,2))
        weight_p = nn.functional.softmax(attention+mask_h, dim=-1)  # b,sp,sh
        weight_h = nn.functional.softmax((attention.transpose(1,2)+mask_p), dim=-1)  # b,sh,sp
        
        attention_p = weight_p.bmm(sent_h)  # b,sp,v
        attention_h = weight_h.bmm(sent_p)  # b,sh,v
        
        return attention_p,attention_h

    # 剔除pad计算的lstm，用于替换原始的lstm
    # 采用了pytorch提供的packed_pad_sequence/pad_packed_sequence 方案
    @classmethod
    def seq2seq(cls,
                input: torch.FloatTensor,  # batch_size,seq_size,feat_size
                mask: torch.ByteTensor,   # batch_size,seq_size
                lstm: nn.LSTM
                ):
        # print('input: ',input)
        # print('mask: ',mask)
        lengths=mask.sum(dim=1)  # seq_size
        # 注意：这里发生了batch内部句子顺序的改变，这是pack_paded_sequence 调用的前提
        # 在过lstm和pad_packed_sequence之后，需要进行恢复，否则无法和标签对应
        lengths,indices=torch.sort(lengths,dim=0,descending=True)
        # print('lengths: ',lengths)
        # print('indices: ',indices)
        sorted_input=input.index_select(0,indices)
        sorted_input_pack=nn.utils.rnn.pack_padded_sequence(sorted_input,lengths,batch_first=True)

        sorted_output_pack,_=lstm(sorted_input_pack) ##

        # sorted_output_pack=sorted_input_pack
        sorted_output,_=torch.nn.utils.rnn.pad_packed_sequence(sorted_output_pack,batch_first=True)

        # 为了恢复原先的顺序，需要将indices扩展为与sorted_output相同维度
        # 扩展1维
        indices=indices.unsqueeze(1).expand(sorted_output.size(0),sorted_output.size(1))
        # 扩展2维
        indices=indices.unsqueeze(2).expand_as(sorted_output)
        # print(indices)

        output=sorted_output.scatter(0,indices,sorted_output)
        # i_size=input.shape
        # o_size=output.shape
        # print('ok')
        # print(i_size,o_size)
        assert input.size(1) == output.size(1)
        return output


    # ESIM 前向计算
    def forward(self,
                batch_inst
                ):
        premise = batch_inst.sent_p  # 1）前提
        hypothesis = batch_inst.sent_h  # 2）假设
        mask_p = batch_inst.mask_p  # 3）前提的mask
        mask_h = batch_inst.mask_h  # 4）假设的mask

        # 1.Embedding 层
        premise_emb = self.embed(premise)
        hypothesis_emb = self.embed(hypothesis)
        if self.has_pretrain:
            premise_ext = batch_inst.sent_p_ext  # 1）前提
            hypothesis_ext = batch_inst.sent_h_ext  # 2）假设
            premise_emb += self.embed_ext(premise_ext)
            hypothesis_emb += self.embed_ext(hypothesis_ext)
        premise, hypothesis = premise_emb, hypothesis_emb

        # 2.Encoding-LSTM
        # premise, _ = self.lstm1(premise)
        # hypothesis, _ = self.lstm1(hypothesis)
        premise = Model.seq2seq(premise,mask_p,self.lstm1)
        hypothesis = Model.seq2seq(hypothesis,mask_h,self.lstm1)
        # print("seq_premise: ",premise.size())
        # print("seq_hypothesis: ",hypothesis.size())

        # 3.Attention
        premise_attention,hypothesis_attention = self.softmax_attention(premise, hypothesis, mask_p, mask_h)

        # 4.整合
        premise = torch.cat([premise, premise_attention, premise-premise_attention, premise*premise_attention], dim=-1)
        hypothesis = torch.cat([hypothesis, hypothesis_attention, hypothesis-hypothesis_attention, hypothesis*hypothesis_attention], dim=-1)

        premise = self.projection(premise)
        hypothesis = self.projection(hypothesis)

        # 5.Composition-LSTM
        premise = Model.seq2seq(premise,mask_p,self.lstm2)
        hypothesis = Model.seq2seq(hypothesis,mask_h,self.lstm2)

        # 6.池化 Pooling
        premise_avg = torch.sum(premise*mask_p.unsqueeze(2).float(), dim=1)/premise.size(1)
        hypothesis_avg = torch.sum(hypothesis*mask_h.unsqueeze(2).float(), dim=1)/hypothesis.size(1)

        premise_max, _ = premise.masked_fill((1-mask_p).unsqueeze(-1).expand_as(premise),-1e4).max(dim=1)
        hypothesis_max, _ = hypothesis.masked_fill((1-mask_h).unsqueeze(-1).expand_as(hypothesis),-1e4).max(dim=1)

        p_h_cat = torch.cat([premise_avg, premise_max, hypothesis_avg, hypothesis_max], dim=1)

        # 7. MLP层 classifier
        logits = self.classifier(p_h_cat)

        probabilities = nn.functional.softmax(logits, dim=-1)

        return probabilities

# 用于模型内部参数（权重，偏置等）的初始化
def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if module.bidirectional:
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0

# 以下测试用于测试seq2seq函数，证实其成功恢复了pad部分和batch中的句子顺序
# if __name__=="__main__":
#     x=torch.autograd.Variable(torch.LongTensor([[3,0,0],[1,2,3],[2,3,0]]))
#     embedding=nn.Embedding(4,2,padding_idx=0)
#     x=embedding(x)
#     m=torch.autograd.Variable(torch.ByteTensor([[1,0,0],[1,1,1],[1,1,0]]))
#     Model.seq2seq(x,m)
