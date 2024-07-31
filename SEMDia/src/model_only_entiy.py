#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from transformers import AutoModel, AutoConfig
from src.common import Triaffine, init_esim_weights, FusionGate, NewFusionGate
# from openhgnn.models import HAN

import torch
import torch.nn as nn
from itertools import accumulate
from transformers.models.t5.modeling_t5 import T5LayerNorm
from einops import rearrange

class BertWordPair(nn.Module):
    def __init__(self, cfg):
        super(BertWordPair, self).__init__()
        self.bert = AutoModel.from_pretrained(cfg.bert_path)
        bert_config = AutoConfig.from_pretrained(cfg.bert_path)
        cfg.hidden_dim = bert_config.hidden_size
        self.totable = TensorcontextSeq2Mat(cfg)
        self.tableencoder = TableEncoder(cfg)
        self.cls_linear_ent = nn.Linear(cfg.inner_dim, 4)
        init_esim_weights(self.totable)
        init_esim_weights(self.tableencoder)
        init_esim_weights(self.cls_linear_ent)


        cfg.out_dim = bert_config.hidden_size
        cfg.num_heads = [cfg.num_head0]

        self.cfg = cfg


    def forward(self, global_epoch=10, **kwargs):
        ###分开计算每一个sentence
        input_ids, input_masks, input_segments, utterance_spans = [kwargs[w] for w in
                                                                   ['input_ids2', 'input_masks2', 'input_segments2',
                                                                    'utterance_spans']]
        bsize, _, sentence_length = input_ids.shape
        input_ids, input_masks, input_segments = input_ids.view(-1, sentence_length), input_masks.view(-1,sentence_length), input_segments.view(-1, sentence_length)
        utterance_spans = utterance_spans.tolist()
        sequence_outputs = self.bert(input_ids, token_type_ids=input_segments, attention_mask=input_masks)[0]


        losses, tags = [], []
        bsize_x, qlen = sequence_outputs.shape[:2]

        table = self.totable(sequence_outputs, sequence_outputs)  # [b,seq,seq,dim]
        table = self.tableencoder(table)
        del (sequence_outputs)
        logits_ent = self.cls_linear_ent(table)
        del (table)

        # get ent
        input_labels = kwargs["ent_matrix"]
        logits = torch.zeros([bsize, input_labels.shape[1], input_labels.shape[1], 4]).to(self.cfg.device)
        input_labels2 = -torch.ones([bsize_x, qlen, qlen]).to(self.cfg.device)
        k = 0
        for i in range(bsize):
            for (a, b) in utterance_spans[i]:
                input_labels2[k, 1:(b - a + 2), 1:(b - a + 2)] = input_labels[i, a:b + 1, a:b + 1]
                logits[i, a:b + 1, a:b + 1, :] = logits_ent[k, 1:(b - a + 2), 1:(b - a + 2), :]
                k += 1

        loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        logits_ent_f = torch.flatten(logits_ent, start_dim=1, end_dim=2)
        input_labels2 = torch.flatten(input_labels2, start_dim=1, end_dim=2)
        loss = loss_func(logits_ent_f.transpose(1, 2), input_labels2.long())
        logits = logits * (kwargs['sentence_masks'][:, :, :, None] > 0)
        losses.append(loss)
        tags.append(logits)

        # 先对entiy进行训练，收敛后再训练relation
        logits2 = torch.zeros([bsize, input_labels.shape[1], input_labels.shape[1], 5]).to(self.cfg.device)
        logits_rel = torch.zeros([bsize, 1, 5]).to(self.cfg.device)
        label_ret = torch.zeros([bsize, 1]).to(self.cfg.device) - 1
        loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        loss2 = loss_func(logits_rel.transpose(1, 2), label_ret.long())
        losses.append(loss2)
        tags.append(logits2)

        return losses, tags



class TensorcontextSeq2Mat(nn.Module):
    """
    refernce: SOCHER R, PERELYGIN A, WU J, 等. Recursive deep models for semantic compositionality over a sentiment treebank[C]//Proceedings of the 2013 conference on empirical methods in natural language processing. 2013: 1631-1642.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d = 64
        hidden_size = config.hidden_size
        self.dense1 = nn.Linear(hidden_size, config.inner_dim)
        self.dense2 = nn.Linear(hidden_size, config.inner_dim)
        self.W = nn.Linear(3 * config.inner_dim + self.d, config.inner_dim)
        self.V = nn.Parameter(torch.Tensor(self.d, config.inner_dim, config.inner_dim))
        self.norm = T5LayerNorm(hidden_size, 1e-12)
        self.activation = nn.GELU()
        self.init_weights()

    def init_weights(self):
        self.V.data.normal_(mean=0.0, std=0.02)

    def rntn(self, x, y, xmat):
        max_len = xmat.shape[1]
        xmat_t = xmat.transpose(1, 2)
        batch_size = xmat.shape[0]
        context = torch.ones_like(x).to(self.config.device)
        for i in range(max_len):
            diag = x.diagonal(dim1=1, dim2=2, offset=-i)
            xmat_t = torch.max(xmat_t[:, :, :max_len - i], diag)
            bb = [[b] for b in range(batch_size)]
            linexup = [[j for j in range(max_len - i)] for b in range(batch_size)]
            lineyup = [[j + i for j in range(max_len - i)] for b in range(batch_size)]
            linexdown = [[j + i for j in range(max_len - i)] for b in range(batch_size)]
            lineydown = [[j for j in range(max_len - i)] for b in range(batch_size)]
            context[bb, linexup, lineyup, :] = xmat_t.permute(0, 2, 1)
            context[bb, linexdown, lineydown, :] = xmat_t.permute(0, 2, 1)

        t = torch.cat([x, y, context], dim=-1)
        xvy = torch.einsum('b m n p, k p d, b m n d -> b m n k', x, self.V, y)
        t = torch.cat([t, xvy], dim=-1)
        tw = self.W(t)
        return tw

    def forward(self, x, y):
        """
        x,y: [B, L, H] => [B, L, L, H]
        """
        x = self.dense1(x)
        y = self.dense2(y)

        xmat = x
        x, y = torch.broadcast_tensors(x[:, :, None], y[:, None, :])
        t = self.rntn(x, y, xmat)
        t = self.activation(t)
        return t

class TableEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = CNNnet(config)

    def forward(self, table):
        table = self.layer(table)
        return table

class CNNnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer_norm_eps = 1e-12
        self.conv1 = nn.Conv2d(
            in_channels=config.inner_dim,
            out_channels=config.inner_dim,
            kernel_size=(3, 3),
            padding=1
        )
        self.norm1 = T5LayerNorm(config.inner_dim, layer_norm_eps)

    def layer_forward(self, x, conv, norm):
        x = conv(x)
        n = x.size(-1)
        x = rearrange(x, 'b d m n -> b (m n) d')
        x = norm(x)
        x = nn.functional.relu(x)
        x = rearrange(x, 'b (m n) d -> b d m n', n=n)
        return x

    def forward(self, x_input, **kwargs):
        x = rearrange(x_input, 'b m n d -> b d m n')
        x = self.layer_forward(x, self.conv1, self.norm1)
        x = rearrange(x, 'b d m n -> b m n d')
        return x
