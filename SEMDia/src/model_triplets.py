#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from transformers import AutoModel, AutoConfig
from src.common import Triaffine, init_esim_weights, FusionGate, NewFusionGate
# from openhgnn.models import HAN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import accumulate
from transformers.models.t5.modeling_t5 import T5LayerNorm
from einops import rearrange

# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data, Batch
from torch_sparse import SparseTensor
from torch_geometric.nn import graclus

from silearn.graph import GraphSparse
from silearn.optimizer.enc.partitioning.propagation import OperatorPropagation
from silearn.model.encoding_tree import Partitioning, EncodingTree
from torch_scatter import scatter_sum




class BertWordPair(nn.Module):
    def __init__(self, cfg):
        super(BertWordPair, self).__init__()
        self.bert = AutoModel.from_pretrained(cfg.bert_path)
        bert_config = AutoConfig.from_pretrained(cfg.bert_path)
        self.v_size = 64
        mlp_size = bert_config.hidden_size
        self.cls_linear_rel = nn.Linear(mlp_size * 3 + self.v_size, 5)
        self.V = nn.Parameter(torch.Tensor(self.v_size, mlp_size, mlp_size))
        init_esim_weights(self.cls_linear_rel)
        init_esim_weights(self.V)
        cfg.hidden_dim = bert_config.hidden_size
        cfg.out_dim = bert_config.hidden_size
        cfg.num_heads = [cfg.num_head0]
        self.cfg = cfg
        self.p = cfg.parallel#0.35
        self.subdia = cfg.types #in {0: global, 1: reply relation, 2: 2D SEM}

    def forward(self, global_epoch=10, **kwargs):
        input_ids, input_masks, input_segments, hgraphs = [kwargs[w] for w in
                                                           ['input_ids', 'input_masks', 'input_segments', 'hgraphs']]
        pred_t, pred_a, pred_o = [kwargs[w] for w in ['pred_t', 'pred_a', 'pred_o']]
        triple = kwargs['triplets'].tolist()
        sequence_outputs = self.bert(input_ids, token_type_ids=input_segments, attention_mask=input_masks)[0]

        losses, tags = [], []
        bsize, qlen = sequence_outputs.shape[:2]

        # 结构熵聚类
        links = kwargs['links']
        gather_links = self.sturct_gether(bsize, links, sequence_outputs, kwargs['utterance_spans'], kwargs["thread_masks"], kwargs["reply_masks"], kwargs['speaker_masks'],
                                          self.p, self.cfg.device)

        xxx, yyy = 0, 0
        for i in range(bsize):
            gather_link = gather_links[i]
            thread_lengths = kwargs['thread_lengths'][i]
            for j in range(1,len(thread_lengths)):
                thread_lengths[j] = thread_lengths[j] + thread_lengths[j-1]
            utterance_spans = kwargs['utterance_spans'][i]
            thread_length = []
            for k in range(1, len(thread_lengths)):
                xx = []
                for j in range(1,len(utterance_spans)):
                    if utterance_spans[j][1] < thread_lengths[k] and utterance_spans[j][0] > thread_lengths[k-1]:
                        xx.append(j)
                xx.append(0)
                thread_length.append(set(xx))

            if len(thread_length) < len(gather_link):
                xxx += 1
            elif len(thread_length) > len(gather_link):
                yyy += 1


        # ent loss
        loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        logits = torch.zeros([bsize, qlen, qlen, 4]).to(self.cfg.device)
        logits_ret = torch.zeros([bsize, 1, 4]).to(self.cfg.device)
        label_ret = torch.zeros([bsize, 1]).to(self.cfg.device)-1
        loss = loss_func(logits_ret.transpose(1, 2), label_ret.long())
        for i in range(bsize):
            for x in pred_t[i]:
                logits[i, x[0], x[1], :] = torch.tensor([0,1,0,0]).to(self.cfg.device)
            for x in pred_a[i]:
                logits[i, x[0], x[1], :] = torch.tensor([0,0,1,0]).to(self.cfg.device)
            for x in pred_o[i]:
                logits[i, x[0], x[1], :] = torch.tensor([0,0,0,1]).to(self.cfg.device)
        losses.append(loss)
        tags.append(logits)

        # relation loss
        loss2, logits2 = self.get_canditity(bsize, pred_t, pred_a, pred_o, kwargs['sentence_masks'], sequence_outputs,
                                            kwargs['rel_matrix'], kwargs['thread_masks'], kwargs['utterance_spans'], gather_links, kwargs['token2sents'], self.cfg.device)
        losses.append(loss2)
        tags.append(logits2)

        return losses, tags, xxx, yyy


    def sturct_gether(self, bsize, links, sequence_outputs, utterance_spans, thread_masks, reply_masks, speaker_masks, p, device):
        max_len = 0
        for i in range(bsize):
            if len(utterance_spans[i]) > max_len:
                max_len = len(utterance_spans[i])
        sentence_r = torch.zeros([bsize, max_len, cfg.hidden_dim])
        for i in range(bsize):
            for k,m in enumerate(utterance_spans[i]):
                # sentence_r[i, k, :] = sequence_outputs[i, m[0]-1, :]
                sentence_r[i, k, :] = torch.max(sequence_outputs[i, m[0]:m[1]+1, :], dim=0)[0]
                # sentence_r[i, k, :] = torch.cat([sequence_outputs[i, m[0]-1, :], torch.max(sequence_outputs[i, m[0]:m[1] + 1, :], dim=0)[0]], dim=-1)

        #构建graph
        adj_matrixs = [[] for i in range(bsize)]
        gather_linkss = []
        for i in range(bsize):
            edge = [[], []]
            weight = []
            mm_len = len(utterance_spans[i])
            for j in range(mm_len):
                for k in range(j, mm_len):
                    if j == k:
                        edge[0].append(j)
                        edge[1].append(j)
                        weight.append(1)
                    s0 = utterance_spans[i][j][0]# + 1
                    e0 = utterance_spans[i][k][0]# + 1
                    if thread_masks[i, s0, e0] == 1:
                        edge[0].append(j)
                        edge[1].append(k)
                        sim = max(torch.cosine_similarity(sentence_r[i, j, :], sentence_r[i, k, :], dim=0), 0)
                        weight.append(sim)
                        edge[0].append(k)
                        edge[1].append(j)
                        weight.append(sim)

            # relation[i] = edge
            # edge_w[i] = weight
            row = torch.tensor(edge[0], dtype=torch.long).to(device)
            col = torch.tensor(edge[1], dtype=torch.long).to(device)
            weight = torch.tensor(weight, dtype=torch.float).to(device)
            adj_matrixs[i] = SparseTensor(row=row, col=col, value=weight)
            adj_matrix = adj_matrixs[i].to_dense().cpu().numpy()

            edges = np.array(adj_matrix.nonzero())  # [2, E]
            ew = adj_matrix[edges[0, :], edges[1, :]]
            ew, edges = torch.tensor(ew, device=device), torch.tensor(edges, device=device).t()
            # ew, edges = torch.tensor(ew), torch.tensor(edges).t()
            dist = scatter_sum(ew, edges[:, 1]) + scatter_sum(ew, edges[:, 0])  # dist/2=di
            dist = dist / (2 * ew.sum())  # ew.sum()=vol(G) dist=di/vol(G)
            # print('construct encoding tree...')
            g = GraphSparse(edges, ew, dist)
            optim = OperatorPropagation(Partitioning(g, None))
            optim.perform(p=p)
            # print('construct encoding tree done')
            division = optim.enc.node_id
            # print(division)

            ##聚类后的结果
            totol_comm = torch.max(division) + 1
            division = division.tolist()
            gather_links = []
            for m in range(totol_comm):
                ll = []
                for k,n in enumerate(division):
                    if k == 0:
                        continue
                    if n == m:
                        ll.append(k)
                if ll != []:
                    gather_links.append(set(ll) | set([0]))
            gather_linkss.append(gather_links)

        return gather_linkss


    def get_canditity(self, bsize, pred_t, pred_a, pred_o, sentence_masks, sequence_outputs, rel_matrix, thread_masks,
                         utterance_spans, gather_links, token2sents, device):

        pairs, label_ret, max_len = self.get_pair_reperstation(bsize, utterance_spans, pred_t, pred_a, pred_o, token2sents, gather_links, rel_matrix, thread_masks, device, sentence_masks)

        # input_ret = torch.zeros([bsize, max_len, cfg.hidden_dim*3+self.v_size]).to(device)
        mask_ret = torch.zeros([bsize, max_len]).to(device)
        mm = torch.zeros([bsize, max_len, cfg.hidden_dim]).to(device)
        nn = torch.zeros([bsize, max_len, cfg.hidden_dim]).to(device)
        context = torch.zeros([bsize, max_len, cfg.hidden_dim]).to(device)
        for i in range(bsize):
            k = 0
            for (s0, s1, e0, e1) in pairs[i]:
                mm[i, k, :] = torch.max(sequence_outputs[i, s0:s1 + 1, :], dim=0)[0]
                nn[i, k, :] = torch.max(sequence_outputs[i, e0:e1 + 1, :], dim=0)[0]
                x0 = min(s0, e0)
                x1 = max(s1, e1)
                context[i, k, :] = torch.max(sequence_outputs[i, x0:x1 + 1, :], dim=0)[0]
                mask_ret[i, k] = 1
                k += 1
        t = torch.cat([mm, nn, context], dim=-1)
        xvy = torch.einsum('b m p, k p d, b m d -> b m k', mm, self.V, nn)
        input_ret = torch.cat([t, xvy], dim=-1)

        #get loss
        logits_rel = self.cls_linear_rel(input_ret)  # [max_len, 5]
        logits = torch.zeros([bsize, sequence_outputs.shape[1], sequence_outputs.shape[1], 5]).to(device)
        for i in range(bsize):
            for k, (s0, s1, e0, e1) in enumerate(pairs[i]):
                if self.cfg.lang == 'zh' and not(s0 > e1 or e0 > s1):
                    continue
                logits[i, s0, e0, :] = logits_rel[i, k, :]

        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_func(logits_rel.transpose(1, 2), label_ret.long())
        return loss, logits


    def get_pair_reperstation(self, bsize, utterance_spans, pred_t, pred_a, pred_o, token2sents, gather_links, rel_matrix, thread_masks, device, sentence_masks):
        pairs = []
        labels = []
        max_len = 0
        subdia = self.subdia

        for i in range(bsize):
            pair = []
            label = []
            length = 0
            for x in pred_t[i]:
                for y in pred_a[i]:
                    mmm = set([token2sents[i][x[0]].item(), token2sents[i][y[0]].item()])
                    if subdia == 0 or (subdia == 1 and thread_masks[i, x[0], y[0]] == 1) or (
                            subdia == 2 and (self.in_gather_links(mmm, gather_links[i]))):
                        pair.append([x[0], x[1], y[0], y[1]])
                        length += 1
                        label.append(rel_matrix[i, x[0], y[0]])
                for y in pred_o[i]:
                    mmm = set([token2sents[i][x[0]].item(), token2sents[i][y[0]].item()])
                    if subdia == 0 or (subdia == 1 and thread_masks[i, x[0], y[0]] == 1) or (
                            subdia == 2 and (self.in_gather_links(mmm, gather_links[i]))):
                        pair.append([x[0], x[1], y[0], y[1]])
                        length += 1
                        label.append(rel_matrix[i, x[0], y[0]])
            for x in pred_a[i]:
                for y in pred_o[i]:
                    mmm = set([token2sents[i][x[0]].item(), token2sents[i][y[0]].item()])
                    if subdia == 0 or (subdia == 1 and thread_masks[i, x[0], y[0]] == 1) or (
                            subdia == 2 and (self.in_gather_links(mmm, gather_links[i]))):
                        pair.append([x[0], x[1], y[0], y[1]])
                        length += 1
                        label.append(rel_matrix[i, x[0], y[0]])
            if length > max_len:
                max_len = length
            pairs.append(pair)
            labels.append(label)

        for i in range(bsize):
            for j in range(len(labels[i]), max_len):
                labels[i].append(-1)
        label_ret = torch.tensor(labels).to(device)  # [b, maxlen]
        return pairs, label_ret, max_len

    def in_gather_links(self, mmm, gather_links):
        x = False
        for i in gather_links:
            if mmm <= i:
                x = True
                break
        return x
