import math

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import precision_score
from torch.nn import init

import torch as th
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from torch.nn import init
from transformers import BertModel, AutoModel


class REConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 num_type=4,
                 weight=True,
                 bias=True,
                 activation=None):
        super(REConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.weight_type = nn.Parameter(th.ones(num_type))

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat, type_info):
        with graph.local_scope():
            aggregate_fn = fn.copy_u('h', 'm')

            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = th.reshape(norm, shp)
                feat = feat * norm

            feat = th.matmul(feat, self.weight)
            graph.srcdata['h'] = feat * self.weight_type[type_info].reshape(-1, 1)
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class AGTLayer(nn.Module):
    def __init__(self, embeddings_dimension, nheads=2, att_dropout=0.5, emb_dropout=0.5, temper=1.0, rl=False, rl_dim=4,
                 beta=1):

        super(AGTLayer, self).__init__()

        self.nheads = nheads
        self.embeddings_dimension = embeddings_dimension

        self.head_dim = self.embeddings_dimension // self.nheads

        self.leaky = nn.LeakyReLU(0.01)

        self.temper = temper

        self.rl_dim = rl_dim

        self.beta = beta

        self.linear_l = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)
        self.linear_r = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)

        self.att_l = nn.Linear(self.head_dim, 1, bias=False)
        self.att_r = nn.Linear(self.head_dim, 1, bias=False)

        if rl:
            self.r_source = nn.Linear(rl_dim, rl_dim * self.nheads, bias=False)
            self.r_target = nn.Linear(rl_dim, rl_dim * self.nheads, bias=False)

        self.linear_final = nn.Linear(
            self.head_dim * self.nheads, self.embeddings_dimension, bias=False)
        self.dropout1 = nn.Dropout(att_dropout)
        self.dropout2 = nn.Dropout(emb_dropout)

        self.LN = nn.LayerNorm(embeddings_dimension)

    def forward(self, h, rh=None):
        batch_size = h.size()[0]
        fl = self.linear_l(h).reshape(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)
        fr = self.linear_r(h).reshape(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)

        score = self.att_l(self.leaky(fl)) + self.att_r(self.leaky(fr)).permute(0, 1, 3, 2)

        if rh is not None:
            r_k = self.r_source(rh).reshape(batch_size, -1, self.nheads, self.rl_dim).transpose(1, 2)
            r_q = self.r_target(rh).reshape(batch_size, -1, self.nheads, self.rl_dim).permute(0, 2, 3, 1)
            score_r = r_k @ r_q
            score = score + self.beta * score_r

        score = score / self.temper

        score = F.softmax(score, dim=-1)
        # print(score.shape)
        score = self.dropout1(score)

        context = score @ fr
        # print(context.shape)
        h_sa = context.transpose(1, 2).reshape(batch_size, -1, self.head_dim * self.nheads)
        fh = self.linear_final(h_sa)
        fh = self.dropout2(fh)
        # print(h.shape, fh.shape)
        h = self.LN(h + fh)

        return h


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / embedding_dim) for i in range(embedding_dim)]
            if pos != 0 else np.zeros(embedding_dim) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])
        self.pos_table = torch.FloatTensor(pos_table).cuda()

    def forward(self, enc_inputs):
        return self.pos_table[:enc_inputs.size(1), :]
        # return self.dropout(enc_inputs.cuda())


class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """

    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=True,
                 bias=False,
                 alpha=0.):
        super(myGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            e_feat = self.edge_emb(e_feat)
            e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
            ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.edata.update({'ee': ee})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e') + graph.edata.pop('ee'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1 - self.alpha) + res_attn * self.alpha
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()


class myGAT(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha):
        super(myGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
                                         num_hidden, num_hidden, heads[0],
                                         feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(edge_dim, num_etypes,
                                             num_hidden * heads[l - 1], num_hidden, heads[l],
                                             feat_drop, attn_drop, negative_slope, residual, self.activation,
                                             alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
                                         num_hidden * heads[-2], num_hidden, heads[-1],
                                         feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()

    def forward(self, features_list, e_feat):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, res_attn=res_attn)
            h = h.flatten(1)
        h, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=None)
        h = h.mean(1)
        return h


class CDSearcher(nn.Module):
    def __init__(self, g, edge_dim, num_etypes, num_class, input_dimensions, embeddings_dimension=64, num_layers=8,
                 num_gnns=2, nheads=2,
                 dropout=0, temper=1.0, num_type=4, beta=1, top_k=5, num_seqs=15):

        super(CDSearcher, self).__init__()

        self.g = g
        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_class = num_class
        self.num_gnns = num_gnns
        self.nheads = nheads
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, embeddings_dimension) for in_dim in input_dimensions])
        self.dropout = dropout
        self.GCNLayers = torch.nn.ModuleList()
        self.RELayers = torch.nn.ModuleList()
        self.QGTLayers = torch.nn.ModuleList()
        self.DGTLayers = torch.nn.ModuleList()
        # self.attr_drop = 0.2
        for layer in range(self.num_gnns * 2):
            # self.GCNLayers.append(GraphConv(
            #     self.embeddings_dimension, self.embeddings_dimension, activation=F.relu, allow_zero_in_degree=True))
            self.RELayers.append(REConv(num_type, num_type, activation=F.relu, num_type=num_type))
        heads = [2] * num_gnns + [1]
        self.GCNLayers = myGAT(self.g, edge_dim, num_etypes, input_dimensions, embeddings_dimension, num_gnns,
                               heads, F.elu, dropout, dropout, 0.05, True, 0.05)
        for layer in range(self.num_layers):
            self.QGTLayers.append(
                AGTLayer(self.embeddings_dimension, self.nheads, self.dropout, self.dropout, temper=temper, rl=True,
                         rl_dim=num_type, beta=beta))
        for layer in range(self.num_layers):
            self.DGTLayers.append(
                AGTLayer(self.embeddings_dimension, self.nheads, self.dropout, self.dropout, temper=temper, rl=True,
                         rl_dim=num_type, beta=beta))
        self.PositionLayer = PositionalEncoding(embeddings_dimension)
        self.Drop = nn.Dropout(self.dropout)
        self.GraphFn = nn.Sequential(
            nn.Linear(embeddings_dimension * 3, embeddings_dimension, bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(embeddings_dimension, embeddings_dimension, bias=True),
        )
        self.TextFn = nn.Sequential(
            nn.Linear(embeddings_dimension * 3, embeddings_dimension, bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(embeddings_dimension, embeddings_dimension, bias=True),
        )

        self.WeightScore1 = nn.Sequential(
            nn.Linear(num_seqs, 1, bias=True),
            nn.LeakyReLU(0.1),
        )
        self.top_k = top_k
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        self.Scores = nn.Sequential(
            nn.Linear(embeddings_dimension + 10 + embeddings_dimension + 1, embeddings_dimension, bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(embeddings_dimension, 1, bias=True),
        )

    def forward(self, features_list, e_feat, qc_seqs, type_emb, node_type, dataset, contexts):
        dc_seqs, mask, dataset_word_ids = dataset
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        r = type_emb[node_type]
        gh = self.GCNLayers(features_list, e_feat)
        qt = h[qc_seqs].max(dim=1).values
        dt = h[dc_seqs[:, 0]].max(dim=1).values
        ct_batch = []
        for context_id in contexts:
            ch = h[torch.tensor(context_id).cuda()]
            ch = torch.unsqueeze(ch.max(dim=0).values, 0)
            ct_batch.append(ch)
        ct = torch.cat(ct_batch, 0)
        qdt = self.TextFn(torch.cat([qt, ct, dt], dim=-1))
        qt = qt.unsqueeze(1)
        dt = dt.unsqueeze(1)
        text_score = torch.matmul(qt, dt.transpose(1, 2)).squeeze(dim=-1)
        for layer in range(self.num_gnns * 2):
            r = self.RELayers[layer](self.g, r, node_type)
        qh = gh[qc_seqs]
        qr = r[qc_seqs]
        for layer in range(self.num_layers):
            qh = self.QGTLayers[layer](qh, rh=qr)
        # qh = qh[:, :, :]

        ch_batch = []
        for context_id in contexts:
            ch = gh[torch.tensor(context_id).cuda()]
            ch = torch.unsqueeze(ch.max(dim=0).values, 0)
            ch_batch.append(ch)
        ch = torch.cat(ch_batch, 0)
        dc_seqs = torch.LongTensor(np.array(dc_seqs))
        dh = gh[dc_seqs]
        dr = r[dc_seqs]
        dh_batch = []
        for s in range(dh.shape[1]):
            new_dh = dh[:, s, :, :]
            new_dh = new_dh + self.PositionLayer(new_dh)
            new_dr = dr[:, s, :, :]
            for layer in range(self.num_layers):
                new_dh = self.DGTLayers[layer](new_dh, rh=new_dr)
            dh_batch.append(new_dh[:, 0, :].unsqueeze(1))
        dh = torch.cat(dh_batch, dim=1)
        qdh = self.GraphFn(torch.cat([qh.max(dim=1).values, ch, dh.max(dim=1).values], dim=-1))
        dh = dh.transpose(1, 2)
        graph_scores = torch.matmul(qh, dh)
        graph_scores = self.WeightScore1(graph_scores)
        graph_scores = graph_scores.squeeze(dim=-1)
        embed = torch.cat([qdh, graph_scores, qdt, text_score], dim=-1)
        embed = F.normalize(embed, dim=-1)
        embed = self.Drop(embed)
        graph_scores = self.Scores(embed)
        graph_scores = graph_scores.squeeze(dim=-1)
        total_scores = graph_scores
        return total_scores
