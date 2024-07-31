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


mlp_size = 256
device = 'cuda:1'

class BertWordPair(nn.Module):
    def __init__(self, cfg):
        super(BertWordPair, self).__init__()
        self.bert = AutoModel.from_pretrained(cfg.bert_path)
        bert_config = AutoConfig.from_pretrained(cfg.bert_path)

        self.totable = TensorSeq2Mat(cfg)
        self.tableencoder = TableEncoder(cfg)
        self.cls_linear_ent = nn.Linear(mlp_size, 4)
        self.cls_linear_rel = nn.Linear(mlp_size*3, 5)
        init_esim_weights(self.totable)
        init_esim_weights(self.tableencoder)
        init_esim_weights(self.cls_linear_ent)
        init_esim_weights(self.cls_linear_rel)
        
        cfg.category = 'tk'

        cfg.meta_paths_dict = {
            'sp-rep': [('tk', 'spk', 'tk'), ('tk', 'rep', 'tk')],
            'rep-sp': [('tk', 'rep', 'tk'), ('tk', 'spk', 'tk')],
            'sp': [('tk', 'spk', 'tk')],
            'rep': [('tk', 'rep', 'tk')],
            'self': [('tk', 'self', 'tk')],
        }
        cfg.hidden_dim = bert_config.hidden_size
        cfg.out_dim = bert_config.hidden_size
        cfg.num_heads = [cfg.num_head0]

        self.cfg = cfg

    def get_loss(self, kwargs, logits, input_labels, mat_name, masks):
        nums = logits.shape[-1]
        # masks = kwargs['sentence_masks'] if mat_name == 'ent' else kwargs['thread_masks']
        criterion = nn.CrossEntropyLoss(logits.new_tensor([1.0] + [self.cfg.loss_weight[mat_name]] * (nums - 1)))

        active_loss = masks.view(-1) == 1
        active_logits = logits.view(-1, logits.shape[-1])[active_loss]
        active_labels = input_labels.view(-1)[active_loss]
        loss = criterion(active_logits, active_labels)

        return loss

    def forward(self, global_epoch=10, **kwargs):
        input_ids, input_masks, input_segments, hgraphs = [kwargs[w] for w in ['input_ids', 'input_masks', 'input_segments', 'hgraphs']]
        sequence_outputs = self.bert(input_ids, token_type_ids=input_segments, attention_mask=input_masks)[0]

        mat_names = ['ent', 'rel']
        losses, tags = [], []
        bsize, qlen = sequence_outputs.shape[:2]

        table = self.totable(sequence_outputs, sequence_outputs)  # [b,seq,seq,dim]
        table = self.tableencoder(table)

        #get ent
        input_labels = kwargs["ent_matrix"]
        logits = self.cls_linear_ent(table)
        loss = self.get_loss(kwargs, logits, input_labels, 'ent', kwargs['sentence_masks'])
        losses.append(loss)
        tags.append(logits)

        # 计算候选entiy。option1：按照thread_masks；option2：设置为结构熵划分的结果。
        thread_masks = kwargs["thread_masks"]
        # 先对entiy进行训练，收敛后再训练relation
        if global_epoch <= 6:
            logits2 = torch.zeros([bsize, table.shape[1], table.shape[1], 5]).to(device)
            logits_rel = torch.zeros([bsize, 1, 5]).to(device)
            label_ret = torch.zeros([bsize, 1]).to(device) - 1
        else:
            pairs, logits2, logits_rel, mask_ret, label_ret = self.get_canditity(bsize, logits, kwargs['sentence_masks'], table, kwargs["rel_matrix"], thread_masks)

        loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        loss2 = loss_func(logits_rel.transpose(1, 2), label_ret.long())
        losses.append(loss2)
        tags.append(logits2)

        return losses, tags

    def get_canditity(self, bsize, logits, sentence_masks, table, rel_matrix, thread_masks):
        # entity_dic = {"O": 0, "ENT-T": 1, "ENT-A": 2, "ENT-O": 3}
        # rel_dic = {"O": 0, "h2h": 1, 'pos': 2, 'neg': 3, 'other': 4}
        pred = logits.argmax(dim=3) * sentence_masks # [b, seq, seq, 1]
        pairs = [[] for i in range(bsize)] #[b, len, []]
        max_len = 0
        for i in range(bsize):
            S_pred = torch.nonzero(pred[i]).cpu().numpy()
            if len(S_pred) > 64:
                S_pred = S_pred[:64]
            for (s0, s1) in S_pred:
                for (e0, e1) in S_pred:
                    if (e0 > s1 or e1 < s0) and thread_masks[i, s0, e1] == 1:  # 上半矩阵，且entiy没有交际
                        pairs[i].append([s0, e0, s1, e1])
            if len(pairs[i]) > max_len:
                max_len = len(pairs[i])

        if max_len == 0:
            logits = torch.zeros([bsize, table.shape[1], table.shape[1], 5]).to(device)
            logits_rel = torch.zeros([bsize, 1, 5]).to(device)
            mask_ret = torch.zeros([bsize, 1]).to(device)
            label_ret = torch.zeros([bsize, 1]).to(device)-1
            return pairs, logits, logits_rel, mask_ret, label_ret

        input_ret = torch.zeros([bsize, max_len, mlp_size * 3]).to(device)
        mask_ret = torch.zeros([bsize, table.shape[1], table.shape[1]]).to(device)
        label_ret = -torch.ones([bsize, max_len]).to(device)
        for i in range(bsize):
            j = 0
            for (s0, e0, s1, e1) in pairs[i]:
                S = table[i, s0, e0, :]
                E = table[i, s1, e1, :]
                R = torch.max(torch.max(table[i, s0:s1+1, e0:e1+1, :], dim=1)[0], dim=0)[0]
                input_ret[i, j, :] = torch.cat([S, E, R])
                mask_ret[i, s0, e0] = 1
                label_ret[i, j] = rel_matrix[i, s0, e0]
                j += 1

        logits_rel = self.cls_linear_rel(input_ret)  # [b, max_len, 5]
        logits = torch.zeros([bsize, table.shape[1], table.shape[1], 5]).to(device)
        for i in range(bsize):
            for j in range(len(pairs[i])):
                (s0, e0, s1, e1) = pairs[i][j]
                logits[i, s0, e0, :] = logits_rel[i, j, :]

        return pairs, logits, logits_rel, mask_ret, label_ret


    def get_canditity_t(self, bsize, logits, sentence_masks, table, rel_matrix, thread_masks):
        # entity_dic = {"O": 0, "ENT-T": 1, "ENT-A": 2, "ENT-O": 3}
        # rel_dic = {"O": 0, "h2h": 1, 'pos': 2, 'neg': 3, 'other': 4}
        pred = logits.argmax(dim=3) * sentence_masks # [b, seq, seq, 1]
        pairs = [[] for i in range(bsize)] #[b, len, []]
        t_pairs = [[] for i in range(bsize)]
        a_pairs = [[] for i in range(bsize)]
        o_pairs = [[] for i in range(bsize)]
        max_len = 0
        for i in range(bsize):
            S_pred = torch.nonzero(pred[i]).cpu().numpy()
            if len(S_pred) > 64:
                S_pred = S_pred[:64]
            for (s0, s1) in S_pred:
                if pred[i, s0, s1] == 1:
                    t_pairs[i].append([s0, s1])
                elif pred[i, s0, s1] == 2:
                    a_pairs[i].append([s0, s1])
                elif pred[i, s0, s1] == 3:
                    o_pairs[i].append([s0, s1])

        for i in range(bsize):
            for (s0, s1) in t_pairs[i]:
                for (e0, e1) in a_pairs[i]:
                    if (s1<e0 or e1<s0) and thread_masks[i, s0, e1] == 1:
                        pairs[i].append([s0, e0, s1, e1])
                for (m0, m1) in o_pairs[i]:
                    if (s1<m0 or m1<s0) and thread_masks[i, s0, m1] == 1:
                        pairs[i].append([s0, m0, s1, m1])
            for (s0, s1) in a_pairs[i]:
                for (m0, m1) in o_pairs[i]:
                    if (s1<m0 or m1<s0) and thread_masks[i, s0, m1] == 1:
                        pairs[i].append([s0, m0, s1, m1])
            if len(pairs[i]) > max_len:
                max_len = len(pairs[i])


        if max_len == 0:
            logits = torch.zeros([bsize, table.shape[1], table.shape[1], 5]).to(device)
            logits_rel = torch.zeros([bsize, 1, 5]).to(device)
            mask_ret = torch.zeros([bsize, 1]).to(device)
            label_ret = torch.zeros([bsize, 1]).to(device)-1
            return pairs, logits, logits_rel, mask_ret, label_ret

        input_ret = torch.zeros([bsize, max_len, mlp_size * 3]).to(device)
        mask_ret = torch.zeros([bsize, table.shape[1], table.shape[1]]).to(device)
        label_ret = -torch.ones([bsize, max_len]).to(device)
        for i in range(bsize):
            j = 0
            for (s0, e0, s1, e1) in pairs[i]:
                S = table[i, s0, e0, :]
                E = table[i, s1, e1, :]
                R = torch.max(torch.max(table[i, s0:s1+1, e0:e1+1, :], dim=1)[0], dim=0)[0]
                input_ret[i, j, :] = torch.cat([S, E, R])
                mask_ret[i, s0, e0] = 1
                label_ret[i, j] = rel_matrix[i, s0, e0]
                j += 1

        logits_rel = self.cls_linear_rel(input_ret)  # [b, max_len, 5]
        logits = torch.zeros([bsize, table.shape[1], table.shape[1], 5]).to('cuda')
        for i in range(bsize):
            for j in range(len(pairs[i])):
                (s0, e0, s1, e1) = pairs[i][j]
                logits[i, s0, e0, :] = logits_rel[i, j, :]

        return pairs, logits, logits_rel, mask_ret, label_ret


class TensorcontextSeq2Mat(nn.Module):
    """
    refernce: SOCHER R, PERELYGIN A, WU J, 等. Recursive deep models for semantic compositionality over a sentiment treebank[C]//Proceedings of the 2013 conference on empirical methods in natural language processing. 2013: 1631-1642.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.h = 12
        self.d = 64
        hidden_size = 1024
        self.dense1 = nn.Linear(hidden_size, mlp_size)
        self.dense2 = nn.Linear(hidden_size, mlp_size)
        self.W = nn.Linear(2 * mlp_size + self.d, mlp_size)
        self.V = nn.Parameter(torch.Tensor(self.d, mlp_size, mlp_size))
        self.norm = T5LayerNorm(hidden_size, 1e-12)
        self.activation = nn.GELU()
        self.init_weights()

    def init_weights(self):
        if self.config.model_type=='bart' or self.config.model_type=='t5':
            self.V.data.normal_(mean=0.0, std=0.02)
        else:
            self.V.data.normal_(mean=0.0, std=self.config.initializer_range)

    def rntn(self, x, y, xmat):
        max_len = xmat.shape[1]
        xmat_t = xmat.transpose(1, 2)
        batch_size = xmat.shape[0]
        context = torch.ones_like(x).to('cuda')
        for i in range(max_len):
            diag = x.diagonal(dim1=1, dim2=2, offset=-i)
            xmat_t = torch.max(xmat_t[:, :, :max_len-i], diag)
            bb = [[b] for b in range(batch_size)]
            linexup = [[j for j in range(max_len-i)] for b in range(batch_size)]
            lineyup = [[j+i for j in range(max_len-i)] for b in range(batch_size)]
            linexdown = [[j+i for j in range(max_len-i)] for b in range(batch_size)]
            lineydown = [[j for j in range(max_len-i)] for b in range(batch_size)]
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

class TensorSeq2Mat(nn.Module):
    """
    refernce: SOCHER R, PERELYGIN A, WU J, 等. Recursive deep models for semantic compositionality over a sentiment treebank[C]//Proceedings of the 2013 conference on empirical methods in natural language processing. 2013: 1631-1642.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.h = 12
        self.d = 64
        hidden_size = 1024
        self.dense1 = nn.Linear(hidden_size, mlp_size)
        self.dense2 = nn.Linear(hidden_size, mlp_size)
        self.W = nn.Linear(2*mlp_size+self.d, mlp_size)
        self.V = nn.Parameter(torch.Tensor(self.d, mlp_size, mlp_size))
        self.norm = T5LayerNorm(hidden_size, 1e-12)
        self.activation = nn.GELU()
        self.init_weights()

    def init_weights(self):
        self.V.data.normal_(mean=0.0, std=0.02)

    def rntn(self, x, y):
        x = self.dense1(x)
        y = self.dense2(y)
        t = torch.cat([x, y], dim=-1)
        xv = torch.einsum('b m n p, k p d -> b m n k d', x, self.V)
        xvy = torch.einsum('b m n k d, b m n d -> b m n k', xv, y)
        t = torch.cat([t, xvy], dim=-1)
        tw = self.W(t)
        return tw

    def forward(self, x, y):
        """
        x,y: [B, L, H] => [B, L, L, H]
        """
        seq = x
        x, y = torch.broadcast_tensors(x[:, :, None], y[:, None, :])
        t = self.rntn(x, y)
        t = self.activation(t)
        return t

class Seq2Mat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.h = 12
        self.d = 64
        hidden_size = 1024
        mlp_size = 256
        self.dense1 = nn.Linear(hidden_size, mlp_size)
        self.dense2 = nn.Linear(hidden_size, mlp_size)
        self.W = nn.Linear(mlp_size*2, mlp_size)
        self.norm = T5LayerNorm(mlp_size, 1e-12)
        self.activation = nn.GELU()

    def forward(self, x, y):
        """
        x,y: [B, L, H] => [B, L, L, H]
        """
        x = self.dense1(x)
        y = self.dense2(y)

        x, y = torch.broadcast_tensors(x[:, :, None], y[:, None, :])
        t = torch.cat([x, y], dim=-1)
        t = self.W(t)
        t = self.activation(t)
        return t


class TableEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_table_layers = 2
        # self.layer = nn.ModuleList([ResNet(config) for _ in range(num_table_layers)])
        self.layer = CNNnet(config)

    def forward(self, table):
        # for layer_module in self.layer:
        #     table = layer_module(table)
        table = self.layer(table)
        return table

class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer_norm_eps = 1e-12
        hidden_size = mlp_size
        self.conv1 = nn.Conv2d(
            in_channels =hidden_size,
            out_channels=hidden_size,
            kernel_size =(1, 1),
            padding=0
        )
        self.norm1 = T5LayerNorm(hidden_size, layer_norm_eps)
        self.conv2 = nn.Conv2d(
            in_channels =hidden_size,
            out_channels=hidden_size,
            kernel_size =(3, 3),
            padding=1
        )
        self.norm2 = T5LayerNorm(hidden_size, layer_norm_eps)

        self.conv3 = nn.Conv2d(
            in_channels =hidden_size,
            out_channels=hidden_size,
            kernel_size =(1, 1),
            padding=0
        )
        self.norm3 = T5LayerNorm(hidden_size, layer_norm_eps)

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
        x = self.layer_forward(x, self.conv2, self.norm2)
        x = self.layer_forward(x, self.conv3, self.norm3)
        x = rearrange(x, 'b d m n -> b m n d')
        return x + x_input

class CNNnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer_norm_eps = 1e-12
        self.conv1 = nn.Conv2d(
            in_channels =mlp_size,
            out_channels=mlp_size,
            kernel_size=(3, 3),
            padding=1
        )
        self.norm1 = T5LayerNorm(mlp_size, layer_norm_eps)

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
