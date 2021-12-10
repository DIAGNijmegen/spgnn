import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from parts import ConvBlock5d, act_wrapper
import torch
import dgl
from utils import topk, get_batch_id
from dgl.nn.pytorch import GATConv, GraphConv, AvgPooling, MaxPooling, SAGEConv, GINConv
import random
import numpy as np
import networkx as nx

def set_trainable(model, trainable):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable


class FeatUNet(nn.Module):
    def __init__(self,
                 n_layers, in_ch_list, base_ch_list, end_ch_list,
                 checkpoint_layers, kernel_sizes,
                 out_ch, padding_list, conv_strides, dropout, spatial_size, fv_dim,
                 norm_method='bn', act_method='relu'):
        super(FeatUNet, self).__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.in_ch_list = in_ch_list
        self.base_ch_list = base_ch_list
        self.checkpoint_layers = checkpoint_layers
        self.conv_strides = conv_strides
        self.spatial_size = spatial_size
        self.kernel_sizes = kernel_sizes
        self.end_ch_list = end_ch_list
        assert (len(end_ch_list) == len(base_ch_list) == len(in_ch_list) == len(padding_list))
        self.out_ch = out_ch
        self.fv_dim = fv_dim
        self.ds_modules = nn.ModuleList(
            [
                ConvBlock5d([in_ch_list[n], base_ch_list[n]], [base_ch_list[n],
                                                               end_ch_list[n]],
                            checkpoint_layers[n], self.kernel_sizes[n], False, padding_list[n],
                            conv_strides=conv_strides[n],
                            norm_method=norm_method, act_method=act_method, dropout=dropout)
                for n in range(n_layers)
            ]
        )
        self.bg = ConvBlock5d([in_ch_list[n_layers], base_ch_list[n_layers]],
                              [base_ch_list[n_layers], end_ch_list[n_layers]],
                              checkpoint_layers[n_layers], self.kernel_sizes[n_layers], False, padding_list[n_layers],
                              dropout=dropout, norm_method=norm_method, act_method=act_method)

        self.fc = nn.Sequential(
            nn.Conv3d(end_ch_list[n_layers], end_ch_list[n_layers], kernel_size=self.spatial_size,
                      padding=0, stride=1, bias=True),
            # normal_wrapper(norm_method, 1024),
            nn.Dropout(dropout),
            act_wrapper(act_method),
            nn.Conv3d(end_ch_list[n_layers], fv_dim, kernel_size=1, padding=0, stride=1, bias=True),
            act_wrapper(act_method)
        )
        self.out = nn.Conv3d(fv_dim, out_channels=out_ch, kernel_size=1, padding=0, bias=True)

    def init(self, initializer):
        initializer.initialize(self)

    def forward(self, x):
        ds_feat_list = [x]
        for idx, ds in enumerate(self.ds_modules):
            if idx == 0:
                ds_feat_list.append(ds(ds_feat_list[-1]))
            else:
                ds_feat_list.append(checkpoint(ds, ds_feat_list[-1]))
        xbg = checkpoint(self.bg, ds_feat_list[-1])
        xbg = self.fc(xbg)
        return self.out(xbg)

    def extract_feature(self, x):
        ds_feat_list = [x]
        for idx, ds in enumerate(self.ds_modules):
            ds_feat_list.append(ds(ds_feat_list[-1]))
        xbg = self.bg(ds_feat_list[-1])
        xbg = self.fc(xbg)
        return xbg, self.out(xbg)


class FeatUNetAddedWeights(nn.Module):
    def __init__(self,
                 n_layers, in_ch_list, base_ch_list, end_ch_list,
                 checkpoint_layers, kernel_sizes,
                 out_ch, padding_list, conv_strides, dropout, spatial_size, fv_dim, added_hiddens,
                 norm_method='bn', act_method='relu'):
        super(FeatUNetAddedWeights, self).__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.in_ch_list = in_ch_list
        self.base_ch_list = base_ch_list
        self.checkpoint_layers = checkpoint_layers
        self.conv_strides = conv_strides
        self.spatial_size = spatial_size
        self.kernel_sizes = kernel_sizes
        self.end_ch_list = end_ch_list
        self.added_hiddens = added_hiddens
        assert (len(end_ch_list) == len(base_ch_list) == len(in_ch_list) == len(padding_list))
        self.out_ch = out_ch
        self.fv_dim = fv_dim
        self.ds_modules = nn.ModuleList(
            [
                ConvBlock5d([in_ch_list[n], base_ch_list[n]], [base_ch_list[n],
                                                               end_ch_list[n]],
                            checkpoint_layers[n], self.kernel_sizes[n], False, padding_list[n],
                            conv_strides=conv_strides[n],
                            norm_method=norm_method, act_method=act_method, dropout=dropout)
                for n in range(n_layers)
            ]
        )
        self.bg = ConvBlock5d([in_ch_list[n_layers], base_ch_list[n_layers]],
                              [base_ch_list[n_layers], end_ch_list[n_layers]],
                              checkpoint_layers[n_layers], self.kernel_sizes[n_layers], False, padding_list[n_layers],
                              dropout=dropout, norm_method=norm_method, act_method=act_method)

        self.fc = nn.Sequential(
            nn.Conv3d(end_ch_list[n_layers], end_ch_list[n_layers], kernel_size=self.spatial_size,
                      padding=0, stride=1, bias=True),
            # normal_wrapper(norm_method, 1024),
            nn.Dropout(dropout),
            act_wrapper(act_method),
            nn.Conv3d(end_ch_list[n_layers], fv_dim, kernel_size=1, padding=0, stride=1, bias=True),
            act_wrapper(act_method)
        )
        self.out = nn.Sequential(*[
            nn.Sequential(nn.Conv3d(h[0], out_channels=h[1], kernel_size=1, padding=0, bias=True),
            nn.Dropout(dropout),
            act_wrapper(act_method)) if ix != len(self.added_hiddens)-1 else
            nn.Conv3d(h[0], out_channels=h[1], kernel_size=1, padding=0, bias=True)
            for ix, h in enumerate(self.added_hiddens)]
        )

    def init(self, initializer):
        initializer.initialize(self)

    def forward(self, x):
        ds_feat_list = [x]
        for idx, ds in enumerate(self.ds_modules):
            if idx == 0:
                ds_feat_list.append(ds(ds_feat_list[-1]))
            else:
                ds_feat_list.append(checkpoint(ds, ds_feat_list[-1]))
        xbg = checkpoint(self.bg, ds_feat_list[-1])
        xbg = self.fc(xbg)
        return self.out(xbg)

    def extract_feature(self, x):
        ds_feat_list = [x]
        for idx, ds in enumerate(self.ds_modules):
            ds_feat_list.append(ds(ds_feat_list[-1]))
        xbg = self.bg(ds_feat_list[-1])
        xbg = self.fc(xbg)
        return xbg, self.out(xbg)

class GCN(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hiddens,
                 num_classes,
                 activation):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        # input projection (no residual)

        self.gcn_layers.append(GraphConv(
            in_dim, num_hiddens[0], activation=activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gcn_layers.append(GraphConv(
                num_hiddens[l - 1], num_hiddens[l], activation=activation))
        # output projection
        self.gcn_layers.append(GraphConv(
            num_hiddens[num_layers - 1],
            num_classes))

    def reset_parameters(self):
        for gcn_layer in self.gcn_layers:
            gcn_layer.reset_parameters()

    def forward(self, g):
        h = g.ndata['fvs']
        for l in range(self.num_layers):
            h = self.gcn_layers[l](g, h)
        # output projection
        logits = self.gcn_layers[-1](g, h)
        return logits

class GCNNet(nn.Module):
    def __init__(self,
                 n_layers, num_gcn_layers,
                 in_ch_list, base_ch_list, end_ch_list,
                 checkpoint_layers, kernel_sizes,
                 out_ch, padding_list, conv_strides, dropout,
                 spatial_size, fv_dim, num_hiddens, node_embed_dim,
                 norm_method='bn', act_method='relu'):
        super(GCNNet, self).__init__()
        self.fv_dim = fv_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.in_ch_list = in_ch_list
        self.base_ch_list = base_ch_list
        self.checkpoint_layers = checkpoint_layers
        self.conv_strides = conv_strides
        self.spatial_size = spatial_size
        self.kernel_sizes = kernel_sizes
        self.num_hiddens = num_hiddens
        self.node_embed_dim = node_embed_dim
        self.end_ch_list = end_ch_list
        assert (len(end_ch_list) == len(base_ch_list) == len(in_ch_list) == len(padding_list))
        self.out_ch = out_ch
        self.ds_modules = nn.ModuleList(
            [
                ConvBlock5d([in_ch_list[n], base_ch_list[n]], [base_ch_list[n],
                                                               end_ch_list[n]],
                            checkpoint_layers[n], self.kernel_sizes[n], False, padding_list[n],
                            conv_strides=conv_strides[n],
                            norm_method=norm_method, act_method=act_method, dropout=dropout)
                for n in range(n_layers)
            ]
        )
        self.bg = ConvBlock5d([in_ch_list[n_layers], base_ch_list[n_layers]],
                              [base_ch_list[n_layers], end_ch_list[n_layers]],
                              checkpoint_layers[n_layers], self.kernel_sizes[n_layers], False, padding_list[n_layers],
                              dropout=dropout, norm_method=norm_method, act_method=act_method)

        self.fc = nn.Sequential(
            nn.Conv3d(end_ch_list[n_layers], end_ch_list[n_layers], kernel_size=self.spatial_size,
                      padding=0, stride=1, bias=True),
            nn.Dropout(dropout),
            act_wrapper(act_method),
            nn.Conv3d(end_ch_list[n_layers], fv_dim, kernel_size=1, padding=0, stride=1, bias=True),
            act_wrapper(act_method)
        )
        self.out = nn.Conv3d(fv_dim, out_channels=out_ch, kernel_size=1, padding=0, bias=True)

        self.gcn = GCN(num_layers=num_gcn_layers, in_dim=fv_dim,
                       num_hiddens=self.num_hiddens, num_classes=node_embed_dim, activation=F.elu)
        self.gnn_out = nn.Linear(node_embed_dim, out_ch)

    def set_gcn_only(self):
        set_trainable(self, False)
        set_trainable(self.gcn, True)
        set_trainable(self.gnn_out, True)

    def init(self, initializer):
        initializer.initialize(self)
        self.gcn.reset_parameters()
        nn.init.xavier_normal_(self.gnn_out.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.constant(self.gnn_out.bias, 0.0)

    def extract_feature(self, x):
        ds_feat_list = [x]
        for idx, ds in enumerate(self.ds_modules):
            ds_feat_list.append(ds(ds_feat_list[-1]))
        xbg = self.bg(ds_feat_list[-1])
        xbg = self.fc(xbg)
        return xbg

    def forward_without_gnn(self, x):
        ds_feat_list = [x]
        for idx, ds in enumerate(self.ds_modules):
            if idx == 0:
                ds_feat_list.append(ds(ds_feat_list[-1]))
            else:
                ds_feat_list.append(checkpoint(ds, ds_feat_list[-1]))
        xbg = self.bg(ds_feat_list[-1])
        xbg = self.fc(xbg)
        return xbg, self.out(xbg)

    def forward(self, g):
        n_embed = self.gcn(g)
        n_out = self.gnn_out(n_embed)
        return n_out, n_embed

class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hiddens,
                 out_ch,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual, norm=False):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.out_ch = out_ch
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hiddens[0], heads[0],
            0.0, 0.0, negative_slope, residual, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hiddens[l - 1] * heads[l - 1], num_hiddens[l], heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hiddens[num_layers - 1] * heads[num_layers - 1],
            out_ch, heads[num_layers],
            0.0, 0.0, negative_slope, residual, None))
        self.norm = norm

    def reset_parameters(self):
        for gat_layer in self.gat_layers:
            gat_layer.reset_parameters()

    def forward(self, g):
        h = g.ndata['fvs']
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        if self.norm:
            logits = F.normalize(logits, p=2, dim=1)
        return logits

    def forward_batch(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.gat_layers, blocks)):
            if l != len(self.gat_layers) - 1:
                h = layer(block, h).flatten(1)
            else:
                h = layer(block, h).mean(1)
                if self.norm:
                    h = F.normalize(h, p=2, dim=1)
        return h


class GIN(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hiddens,
                 out_ch,
                 norm=False):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.gin_layers = nn.ModuleList()
        self.in_dim = in_dim
        self.out_ch = out_ch
        # hidden layers
        for l in range(num_layers):
            if l == 0:
                self.gin_layers.append(GINConv(
                    nn.Sequential(
                        nn.Linear(in_dim, num_hiddens[l]),
                        nn.Dropout(0.1),
                        nn.LeakyReLU(),
                        nn.Linear(num_hiddens[l], num_hiddens[l]),
                        nn.LeakyReLU(),
                    ), "mean", learn_eps=True))
            else:
                self.gin_layers.append(GINConv(
                    nn.Sequential(
                        nn.Linear(num_hiddens[l-1], num_hiddens[l]),
                        nn.Dropout(0.1),
                        nn.LeakyReLU(),
                        nn.Linear(num_hiddens[l], num_hiddens[l]),
                        nn.LeakyReLU(),
                    ), "mean", learn_eps=True))
        # output projection
        self.gin_layers.append(GINConv(
            nn.Sequential(
                nn.Linear(num_hiddens[num_layers - 1], out_ch),
                nn.Dropout(0.1),
                nn.LeakyReLU(),
                nn.Linear(out_ch, out_ch),
                nn.LeakyReLU(),
            ), "mean", learn_eps=True))
        self.norm = norm

    def forward(self, g):
        h = g.ndata['fvs']
        for layer in self.gin_layers:
            h = layer(g, h)
        if self.norm:
            h = F.normalize(h, p=2, dim=1)
        return h

    def forward_batch(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.gin_layers, blocks)):
            h = layer(block, h)
        if self.norm:
            h = F.normalize(h, p=2, dim=1)
        return h


class GATPSPGNN(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 pos_in_dim,
                 num_hiddens,
                 pos_hiddens,
                 pos_heads,
                 out_ch,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual, norm=False, p_activation=F.tanh):
        super(GATPSPGNN, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.pos_hiddens = pos_hiddens
        self.out_ch = out_ch
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim + pos_in_dim, num_hiddens[0], heads[0],
            0.0, 0.0, negative_slope, residual, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hiddens[l - 1] * heads[l - 1] + pos_hiddens[l - 1] * pos_heads[l - 1],
                num_hiddens[l], heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hiddens[num_layers - 1] * heads[num_layers - 1]
            + pos_hiddens[num_layers - 1] * pos_heads[num_layers - 1],
            out_ch, heads[num_layers],
            0.0, 0.0, negative_slope, residual, self.activation))

        self.pgnn_layers = nn.ModuleList()
        self.pgnn_layers.append(GATConv(
            pos_in_dim, pos_hiddens[0], pos_heads[0],
            0.0, 0.0, negative_slope, True, p_activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            if l == num_layers - 1:
                self.pgnn_layers.append(GATConv(
                    pos_hiddens[l - 1] * pos_heads[l - 1], pos_hiddens[l], pos_heads[l],
                    0.0, 0.0, negative_slope, True, p_activation))
            else:
                self.pgnn_layers.append(GATConv(
                    pos_hiddens[l - 1] * pos_heads[l - 1], pos_hiddens[l], pos_heads[l],
                    feat_drop, attn_drop, negative_slope, True, p_activation))
        # # output projection
        # self.pgnn_layers.append(GATConv(
        #     pos_hiddens[num_layers - 1] * pos_heads[num_layers - 1],
        #     out_pos_ch, pos_heads[num_layers],
        #     0.0, 0.0, negative_slope, True, F.tanh))

        self.norm = norm

    def reset_parameters(self):
        for gat_layer in self.gat_layers:
            gat_layer.reset_parameters()

        for gat_layer in self.pgnn_layers:
            gat_layer.reset_parameters()

    def forward(self, g):
        # forward pos
        h_p = g.ndata['pos_enc']
        h_s = g.ndata['fvs']
        for l in range(self.num_layers):
            h_s = torch.cat([h_s, h_p], dim=1)
            h_s = self.gat_layers[l](g, h_s).flatten(1)
            h_p = self.pgnn_layers[l](g, h_p).flatten(1)

        h_s = torch.cat([h_s, h_p], dim=1)
        h_s = self.gat_layers[-1](g, h_s).mean(1)

        return h_s, h_p


class GATPSPGNNNL(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 pos_in_dim,
                 num_hiddens,
                 out_ch,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual, norm=False, ):
        super(GATPSPGNNNL, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.out_ch = out_ch
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim + pos_in_dim, num_hiddens[0], heads[0],
            0.0, 0.0, negative_slope, residual, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hiddens[l - 1] * heads[l - 1] + pos_in_dim,
                num_hiddens[l], heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hiddens[num_layers - 1] * heads[num_layers - 1]
            + pos_in_dim,
            out_ch, heads[num_layers],
            0.0, 0.0, negative_slope, residual, self.activation))
        self.norm = norm

    def reset_parameters(self):
        for gat_layer in self.gat_layers:
            gat_layer.reset_parameters()


    def forward(self, g):
        # forward pos
        h_p = g.ndata['pos_enc']
        h_s = g.ndata['fvs']
        for l in range(self.num_layers):
            h_s = torch.cat([h_s, h_p], dim=1)
            h_s = self.gat_layers[l](g, h_s).flatten(1)

        h_s = torch.cat([h_s, h_p], dim=1)
        h_s = self.gat_layers[-1](g, h_s).mean(1)

        return h_s, h_p


# class GCN(nn.Module):
#     def __init__(self,
#                  num_layers,
#                  in_dim,
#                  num_hiddens,
#                  num_classes):
#         super(GCN, self).__init__()
#         self.num_layers = num_layers
#         self.gat_layers = nn.ModuleList()
#         # input projection (no residual)
#
#         self.gat_layers.append(GraphConv(
#             in_dim, num_hiddens[0]))
#         # hidden layers
#         for l in range(1, num_layers):
#             # due to multi-head, the in_dim = num_hidden * num_heads
#             self.gat_layers.append(GraphConv(
#                 num_hiddens[l - 1], num_hiddens[l]))
#         # output projection
#         self.gat_layers.append(GraphConv(
#             num_hiddens[num_layers - 1],
#             num_classes))
#
#     def reset_parameters(self):
#         for gat_layer in self.gat_layers:
#             gat_layer.reset_parameters()
#
#     def forward(self, g):
#         h = g.ndata['fvs']
#         for l in range(self.num_layers):
#             h = self.gat_layers[l](g, h)
#         # output projection
#         logits = self.gat_layers[-1](g, h)
#         return logits


# class SAGE(nn.Module):
#     def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
#         super().__init__()
#         self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)
#
#     def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.n_classes = n_classes
#         self.layers = nn.ModuleList()
#         if n_layers > 1:
#             self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
#             for i in range(1, n_layers - 1):
#                 self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
#             self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
#         else:
#             self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))
#         self.dropout = nn.Dropout(dropout)
#         self.activation = activation
#
#     def forward(self, blocks, x):
#         h = x
#         for l, (layer, block) in enumerate(zip(self.layers, blocks)):
#             h = layer(block, h)
#             if l != len(self.layers) - 1:
#                 h = self.activation(h)
#                 h = self.dropout(h)
#         return h
#
#     def inference(self, g, x, device, batch_size, num_workers):
#         """
#         Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
#         g : the entire graph.
#         x : the input of entire node set.
#         The inference code is written in a fashion that it could handle any number of nodes and
#         layers.
#         """
#         # During inference with sampling, multi-layer blocks are very inefficient because
#         # lots of computations in the first few layers are repeated.
#         # Therefore, we compute the representation of all nodes layer by layer.  The nodes
#         # on each layer are of course splitted in batches.
#         # TODO: can we standardize this?
#         for l, layer in enumerate(self.layers):
#             y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)
#
#             sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
#             dataloader = dgl.dataloading.NodeDataLoader(
#                 g,
#                 th.arange(g.num_nodes()).to(g.device),
#                 sampler,
#                 device=device if num_workers == 0 else None,
#                 batch_size=batch_size,
#                 shuffle=False,
#                 drop_last=False,
#                 num_workers=num_workers)
#
#             for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
#                 block = blocks[0]
#
#                 block = block.int().to(device)
#                 h = x[input_nodes].to(device)
#                 h = layer(block, h)
#                 if l != len(self.layers) - 1:
#                     h = self.activation(h)
#                     h = self.dropout(h)
#
#                 y[output_nodes] = h.cpu()
#
#             x = y
#         return y

class SAGE(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hiddens,
                 out_ch,
                 node_ks,
                 node_sample_rate=0.3,
                 activation=F.elu,
                 feat_drop=0.1,
                 aggregator_type="pool",
                 norm=None):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.g_layers = nn.ModuleList()
        self.node_ks = node_ks
        self.node_sample_rate = node_sample_rate
        self.out_ch = out_ch
        self.g_layers.append(SAGEConv(
            in_dim, num_hiddens[0], aggregator_type=aggregator_type,
            feat_drop=0.0, activation=activation, norm=norm))
        # hidden layers
        for l in range(1, num_layers):
            self.g_layers.append(SAGEConv(
                num_hiddens[l - 1], num_hiddens[l], aggregator_type=aggregator_type,
                feat_drop=feat_drop, activation=activation, norm=norm))
        # output projection
        self.g_layers.append(SAGEConv(
            num_hiddens[num_layers - 1],
            out_ch, aggregator_type=aggregator_type))

    def reset_parameters(self):
        for g_layer in self.g_layers:
            g_layer.reset_parameters()

    def forward_batch(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.g_layers, blocks)):
            h = layer(block, h)
        return h

    def forward(self, g):
        h = g.ndata['fvs']
        for layer in self.g_layers:
            h = layer(g, h)

        return h

    # def forward(self, g):
    #     device = g.device
    #     x = g.ndata['fvs']
    #     for l, layer in enumerate(self.g_layers):
    #         y = torch.zeros(g.num_nodes(), layer._out_feats).cuda()
    #
    #         sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    #         dataloader = dgl.dataloading.NodeDataLoader(
    #             g,
    #             torch.arange(g.num_nodes()).to(g.device),
    #             sampler,
    #             device=device,
    #             batch_size=g.num_nodes() // 2,
    #             shuffle=False,
    #             drop_last=False,
    #             num_workers=0)
    #
    #         for input_nodes, output_nodes, blocks in dataloader:
    #             block = blocks[0]
    #
    #             block = block.int().to(device)
    #             h = x[input_nodes].to(device)
    #             h = layer(block, h)
    #             y[output_nodes] = h
    #         x = y
    #     return y

class SAGENet(nn.Module):
    def __init__(self,
                 n_layers, num_layers,
                 in_ch_list, base_ch_list, end_ch_list,
                 checkpoint_layers, kernel_sizes,
                 out_ch, padding_list, conv_strides, dropout, feat_drop,
                 spatial_size, fv_dim, num_hiddens, node_embed_dim, node_ks, node_sample_rate,
                 aggregator_type='pool',
                 norm_method='bn', act_method='relu'):
        super(SAGENet, self).__init__()
        self.fv_dim = fv_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.in_ch_list = in_ch_list
        self.base_ch_list = base_ch_list
        self.checkpoint_layers = checkpoint_layers
        self.node_ks = node_ks
        self.num_layers = num_layers
        self.node_sample_rate = node_sample_rate
        self.aggregator_type = aggregator_type
        self.conv_strides = conv_strides
        self.spatial_size = spatial_size
        self.kernel_sizes = kernel_sizes
        self.num_hiddens = num_hiddens
        self.node_embed_dim = node_embed_dim
        self.end_ch_list = end_ch_list
        assert (len(end_ch_list) == len(base_ch_list) == len(in_ch_list) == len(padding_list))
        self.out_ch = out_ch
        self.ds_modules = nn.ModuleList(
            [
                ConvBlock5d([in_ch_list[n], base_ch_list[n]], [base_ch_list[n],
                                                               end_ch_list[n]],
                            checkpoint_layers[n], self.kernel_sizes[n], False, padding_list[n],
                            conv_strides=conv_strides[n],
                            norm_method=norm_method, act_method=act_method, dropout=dropout)
                for n in range(n_layers)
            ]
        )
        self.bg = ConvBlock5d([in_ch_list[n_layers], base_ch_list[n_layers]],
                              [base_ch_list[n_layers], end_ch_list[n_layers]],
                              checkpoint_layers[n_layers], self.kernel_sizes[n_layers], False, padding_list[n_layers],
                              dropout=dropout, norm_method=norm_method, act_method=act_method)

        self.fc = nn.Sequential(
            nn.Conv3d(end_ch_list[n_layers], end_ch_list[n_layers], kernel_size=self.spatial_size,
                      padding=0, stride=1, bias=True),
            nn.Dropout(dropout),
            act_wrapper(act_method),
            nn.Conv3d(end_ch_list[n_layers], fv_dim, kernel_size=1, padding=0, stride=1, bias=True),
            act_wrapper(act_method)
        )
        self.out = nn.Conv3d(fv_dim, out_channels=out_ch, kernel_size=1, padding=0, bias=True)

        self.sage = SAGE(num_layers=num_layers, in_dim=fv_dim,
                       num_hiddens=self.num_hiddens, out_ch=node_embed_dim,
                       activation=F.elu, feat_drop=feat_drop,  node_ks=node_ks,
                       aggregator_type=aggregator_type, node_sample_rate=node_sample_rate)
        self.gnn_out = nn.Linear(node_embed_dim, out_ch)

    def set_gcn_only(self):
        set_trainable(self, False)
        set_trainable(self.sage, True)
        set_trainable(self.gnn_out, True)

    def init(self, initializer):
        initializer.initialize(self)
        self.sage.reset_parameters()
        nn.init.xavier_normal_(self.gnn_out.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.constant(self.gnn_out.bias, 0.0)

    def extract_feature(self, x):
        ds_feat_list = [x]
        for idx, ds in enumerate(self.ds_modules):
            ds_feat_list.append(ds(ds_feat_list[-1]))
        xbg = self.bg(ds_feat_list[-1])
        xbg = self.fc(xbg)
        return xbg

    def forward_without_gnn(self, x):
        ds_feat_list = [x]
        for idx, ds in enumerate(self.ds_modules):
            if idx == 0:
                ds_feat_list.append(ds(ds_feat_list[-1]))
            else:
                ds_feat_list.append(checkpoint(ds, ds_feat_list[-1]))
        xbg = self.bg(ds_feat_list[-1])
        xbg = self.fc(xbg)
        return xbg, self.out(xbg)

    def forward_batch(self, blocks, x):
        n_embed = self.sage.forward_batch(blocks, x)
        n_out = self.gnn_out(n_embed)
        return n_out, n_embed

    def forward(self, g):
        n_embed = self.sage.forward(g)
        n_out = self.gnn_out(n_embed)
        return n_out, n_embed

class GATNet(nn.Module):
    def __init__(self,
                 n_layers, num_gat_layers, num_heads, num_out_heads,
                 in_ch_list, base_ch_list, end_ch_list,
                 checkpoint_layers, kernel_sizes,
                 out_ch, padding_list, conv_strides, dropout, feat_drop, attn_drop, negative_slope,
                 spatial_size, fv_dim, num_hiddens, node_embed_dim, res=True,
                 norm_method='bn', act_method='relu'):
        super(GATNet, self).__init__()
        self.fv_dim = fv_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.in_ch_list = in_ch_list
        self.res = res
        self.base_ch_list = base_ch_list
        self.checkpoint_layers = checkpoint_layers
        self.conv_strides = conv_strides
        self.spatial_size = spatial_size
        self.kernel_sizes = kernel_sizes
        self.num_hiddens = num_hiddens
        self.node_embed_dim = node_embed_dim
        self.end_ch_list = end_ch_list
        assert (len(end_ch_list) == len(base_ch_list) == len(in_ch_list) == len(padding_list))
        self.out_ch = out_ch
        self.ds_modules = nn.ModuleList(
            [
                ConvBlock5d([in_ch_list[n], base_ch_list[n]], [base_ch_list[n],
                                                               end_ch_list[n]],
                            checkpoint_layers[n], self.kernel_sizes[n], False, padding_list[n],
                            conv_strides=conv_strides[n],
                            norm_method=norm_method, act_method=act_method, dropout=dropout)
                for n in range(n_layers)
            ]
        )
        self.bg = ConvBlock5d([in_ch_list[n_layers], base_ch_list[n_layers]],
                              [base_ch_list[n_layers], end_ch_list[n_layers]],
                              checkpoint_layers[n_layers], self.kernel_sizes[n_layers], False, padding_list[n_layers],
                              dropout=dropout, norm_method=norm_method, act_method=act_method)

        self.fc = nn.Sequential(
            nn.Conv3d(end_ch_list[n_layers], end_ch_list[n_layers], kernel_size=self.spatial_size,
                      padding=0, stride=1, bias=True),
            nn.Dropout(dropout),
            act_wrapper(act_method),
            nn.Conv3d(end_ch_list[n_layers], fv_dim, kernel_size=1, padding=0, stride=1, bias=True),
            act_wrapper(act_method)
        )
        self.out = nn.Conv3d(fv_dim, out_channels=out_ch, kernel_size=1, padding=0, bias=True)
        heads = [num_heads] * num_gat_layers + [num_out_heads]
        self.gat = GAT(num_layers=num_gat_layers, in_dim=fv_dim,
                       num_hiddens=self.num_hiddens, out_ch=node_embed_dim, heads=heads,
                       activation=F.elu, feat_drop=feat_drop, attn_drop=attn_drop,
                       negative_slope=negative_slope, residual=self.res)
        self.gnn_out = nn.Linear(node_embed_dim, out_ch)


    def set_gcn_only(self):
        set_trainable(self, False)
        set_trainable(self.gat, True)
        set_trainable(self.gnn_out, True)


    def set_cnn_only(self):
        set_trainable(self, False)
        set_trainable(self.ds_modules, True)
        set_trainable(self.bg, True)
        set_trainable(self.fc, True)
        set_trainable(self.out, True)

    def set_all(self):
        set_trainable(self, True)

    def init(self, initializer):
        initializer.initialize(self)
        self.gat.reset_parameters()
        nn.init.xavier_normal_(self.gnn_out.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.constant(self.gnn_out.bias, 0.0)

    def extract_feature(self, x):
        ds_feat_list = [x]
        for idx, ds in enumerate(self.ds_modules):
            ds_feat_list.append(ds(ds_feat_list[-1]))
        xbg = self.bg(ds_feat_list[-1])
        xbg = self.fc(xbg)
        return xbg

    def forward_without_gnn(self, x):
        ds_feat_list = [x]
        for idx, ds in enumerate(self.ds_modules):
            if idx == 0:
                ds_feat_list.append(ds(ds_feat_list[-1]))
            else:
                ds_feat_list.append(checkpoint(ds, ds_feat_list[-1]))
        xbg = self.bg(ds_feat_list[-1])
        xbg = self.fc(xbg)
        return xbg, self.out(xbg)

    def forward(self, g):
        n_embed = self.gat(g)
        n_out = self.gnn_out(n_embed)
        return n_out, n_embed

    def forward_emb(self, g):
        n_embed = self.gat(g)
        return n_embed, n_embed

    def forward_batch(self, blocks, x):
        n_embed = self.gat.forward_batch(blocks, x)
        n_out = self.gnn_out(n_embed)
        return n_out, n_embed


class GINNet(nn.Module):
    def __init__(self,
                 n_layers, num_gin_layers,
                 in_ch_list, base_ch_list, end_ch_list,
                 checkpoint_layers, kernel_sizes,
                 out_ch, padding_list, conv_strides, dropout,
                 spatial_size, fv_dim, num_hiddens, node_embed_dim,
                 norm_method='bn', act_method='relu'):
        super(GINNet, self).__init__()
        self.fv_dim = fv_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.in_ch_list = in_ch_list
        self.base_ch_list = base_ch_list
        self.checkpoint_layers = checkpoint_layers
        self.conv_strides = conv_strides
        self.spatial_size = spatial_size
        self.kernel_sizes = kernel_sizes
        self.num_hiddens = num_hiddens
        self.node_embed_dim = node_embed_dim
        self.end_ch_list = end_ch_list
        assert (len(end_ch_list) == len(base_ch_list) == len(in_ch_list) == len(padding_list))
        self.out_ch = out_ch
        self.ds_modules = nn.ModuleList(
            [
                ConvBlock5d([in_ch_list[n], base_ch_list[n]], [base_ch_list[n],
                                                               end_ch_list[n]],
                            checkpoint_layers[n], self.kernel_sizes[n], False, padding_list[n],
                            conv_strides=conv_strides[n],
                            norm_method=norm_method, act_method=act_method, dropout=dropout)
                for n in range(n_layers)
            ]
        )
        self.bg = ConvBlock5d([in_ch_list[n_layers], base_ch_list[n_layers]],
                              [base_ch_list[n_layers], end_ch_list[n_layers]],
                              checkpoint_layers[n_layers], self.kernel_sizes[n_layers], False, padding_list[n_layers],
                              dropout=dropout, norm_method=norm_method, act_method=act_method)

        self.fc = nn.Sequential(
            nn.Conv3d(end_ch_list[n_layers], end_ch_list[n_layers], kernel_size=self.spatial_size,
                      padding=0, stride=1, bias=True),
            nn.Dropout(dropout),
            act_wrapper(act_method),
            nn.Conv3d(end_ch_list[n_layers], fv_dim, kernel_size=1, padding=0, stride=1, bias=True),
            act_wrapper(act_method)
        )
        self.out = nn.Conv3d(fv_dim, out_channels=out_ch, kernel_size=1, padding=0, bias=True)

        self.gin = GIN(num_layers=num_gin_layers, in_dim=fv_dim,
                       num_hiddens=self.num_hiddens, out_ch=node_embed_dim,
                       )
        self.gnn_out = nn.Linear(node_embed_dim, out_ch)
        self.gnn_lobe_out = nn.Linear(node_embed_dim, 6)
        self.gnn_lung_out = nn.Linear(node_embed_dim, 3)

    def set_gcn_only(self):
        set_trainable(self, False)
        set_trainable(self.gin, True)
        set_trainable(self.gnn_out, True)
        set_trainable(self.gnn_lobe_out, True)
        set_trainable(self.gnn_lung_out, True)

    def set_cnn_only(self):
        set_trainable(self, False)
        set_trainable(self.ds_modules, True)
        set_trainable(self.bg, True)
        set_trainable(self.fc, True)
        set_trainable(self.out, True)

    def set_all(self):
        set_trainable(self, True)

    def init(self, initializer):
        initializer.initialize(self)
        nn.init.xavier_normal_(self.gnn_out.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.constant(self.gnn_out.bias, 0.0)

    def extract_feature(self, x):
        ds_feat_list = [x]
        for idx, ds in enumerate(self.ds_modules):
            ds_feat_list.append(ds(ds_feat_list[-1]))
        xbg = self.bg(ds_feat_list[-1])
        xbg = self.fc(xbg)
        return xbg

    def forward_without_gnn(self, x):
        ds_feat_list = [x]
        for idx, ds in enumerate(self.ds_modules):
            if idx == 0:
                ds_feat_list.append(ds(ds_feat_list[-1]))
            else:
                ds_feat_list.append(checkpoint(ds, ds_feat_list[-1]))
        xbg = self.bg(ds_feat_list[-1])
        xbg = self.fc(xbg)
        return xbg, self.out(xbg)

    def forward(self, g):
        n_embed = self.gin(g)
        n_out = self.gnn_out(n_embed)
        return n_out, n_embed

    def forward_batch(self, blocks, x):
        n_embed = self.gin.forward_batch(blocks, x)
        n_out = self.gnn_out(n_embed)
        return n_out, n_embed

    def forward_all(self, g):
        n_embed = self.gin(g)
        n_out = self.gnn_out(n_embed)
        n_lobe_out = self.gnn_lobe_out(n_embed)
        n_lung_out = self.gnn_lung_out(n_embed)
        return n_out, n_lobe_out, n_lung_out, n_embed


class GATPositionSPGNNNet(nn.Module):
    def __init__(self,
                 n_layers, num_gat_layers, num_heads, num_out_heads,
                 in_ch_list, base_ch_list, end_ch_list,
                 checkpoint_layers, kernel_sizes,
                 out_ch, padding_list, conv_strides, dropout, feat_drop, attn_drop, negative_slope,
                 spatial_size, fv_dim, num_hiddens, pos_hiddens, num_pos_heads, node_embed_dim, pos_enc_dim,
                 encodng_merge='cat', norm=False, res=True,
                 norm_method='bn', act_method='relu', p_act="tahn", mode="PEL"):
        super(GATPositionSPGNNNet, self).__init__()
        self.fv_dim = fv_dim
        self.dropout = dropout
        self.pos_enc_dim = pos_enc_dim
        self.num_pos_heads = num_pos_heads
        self.encodng_merge = encodng_merge
        self.n_layers = n_layers
        self.res = res
        if p_act == "tahn":
            self.p_act = F.tanh
        else:
            self.p_act = F.elu
        self.mode = mode
        self.in_ch_list = in_ch_list
        self.base_ch_list = base_ch_list
        self.checkpoint_layers = checkpoint_layers
        self.conv_strides = conv_strides
        self.spatial_size = spatial_size
        self.kernel_sizes = kernel_sizes
        self.num_hiddens = num_hiddens
        self.pos_hiddens = pos_hiddens
        self.node_embed_dim = node_embed_dim
        self.end_ch_list = end_ch_list
        print(f"gat norm> ? {norm}.....")
        assert (len(end_ch_list) == len(base_ch_list) == len(in_ch_list) == len(padding_list))
        self.out_ch = out_ch
        self.ds_modules = nn.ModuleList(
            [
                ConvBlock5d([in_ch_list[n], base_ch_list[n]], [base_ch_list[n],
                                                               end_ch_list[n]],
                            checkpoint_layers[n], self.kernel_sizes[n], False, padding_list[n],
                            conv_strides=conv_strides[n],
                            norm_method=norm_method, act_method=act_method, dropout=dropout)
                for n in range(n_layers)
            ]
        )
        self.bg = ConvBlock5d([in_ch_list[n_layers], base_ch_list[n_layers]],
                              [base_ch_list[n_layers], end_ch_list[n_layers]],
                              checkpoint_layers[n_layers], self.kernel_sizes[n_layers], False, padding_list[n_layers],
                              dropout=dropout, norm_method=norm_method, act_method=act_method)

        self.fc = nn.Sequential(
            nn.Conv3d(end_ch_list[n_layers], end_ch_list[n_layers], kernel_size=self.spatial_size,
                      padding=0, stride=1, bias=True),
            nn.Dropout(dropout),
            act_wrapper(act_method),
            nn.Conv3d(end_ch_list[n_layers], fv_dim, kernel_size=1, padding=0, stride=1, bias=True),
            act_wrapper(act_method)
        )
        self.out = nn.Conv3d(fv_dim, out_channels=out_ch, kernel_size=1, padding=0, bias=True)
        heads = [num_heads] * num_gat_layers + [num_out_heads]
        pos_heads = [num_pos_heads] * num_gat_layers + [num_pos_heads]
        if self.mode == "PEL":
            self.gat = GATPSPGNN(num_layers=num_gat_layers, in_dim=fv_dim, pos_in_dim=pos_enc_dim,
                                 num_hiddens=self.num_hiddens, pos_hiddens=pos_hiddens, pos_heads=pos_heads,
                                 out_ch=node_embed_dim, heads=heads,
                                 activation=F.elu, feat_drop=feat_drop, attn_drop=attn_drop,
                                 negative_slope=negative_slope, residual=self.res, norm=norm,
                                 p_activation=self.p_act)
        elif self.mode == "PENL":
            self.gat = GATPSPGNNNL(num_layers=num_gat_layers, in_dim=fv_dim, pos_in_dim=pos_enc_dim,
                                   num_hiddens=self.num_hiddens,
                                   out_ch=node_embed_dim, heads=heads,
                                   activation=F.elu, feat_drop=feat_drop, attn_drop=attn_drop,
                                   negative_slope=negative_slope, residual=self.res, norm=norm)

        self.gnn_out = nn.Linear(node_embed_dim, out_ch)

    def set_gcn_only(self):
        set_trainable(self, False)
        set_trainable(self.gat, True)
        set_trainable(self.gnn_out, True)

    def set_cnn_only(self):
        set_trainable(self, False)
        set_trainable(self.ds_modules, True)
        set_trainable(self.bg, True)
        set_trainable(self.fc, True)
        set_trainable(self.out, True)

    def set_all(self):
        set_trainable(self, True)

    def init(self, initializer):
        initializer.initialize(self)
        self.gat.reset_parameters()
        nn.init.xavier_normal_(self.gnn_out.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.constant(self.gnn_out.bias, 0.0)

    def extract_feature(self, x):
        ds_feat_list = [x]
        for idx, ds in enumerate(self.ds_modules):
            ds_feat_list.append(ds(ds_feat_list[-1]))
        xbg = self.bg(ds_feat_list[-1])
        xbg = self.fc(xbg)
        return xbg

    def forward_without_gnn(self, x):
        ds_feat_list = [x]
        for idx, ds in enumerate(self.ds_modules):
            if idx == 0:
                ds_feat_list.append(ds(ds_feat_list[-1]))
            else:
                ds_feat_list.append(checkpoint(ds, ds_feat_list[-1]))
        xbg = self.bg(ds_feat_list[-1])
        xbg = self.fc(xbg)
        return xbg, self.out(xbg)

    def forward(self, g):
        n_embed, n_p_embed = self.gat(g)
        n_out = self.gnn_out(n_embed)
        return n_out, n_embed, n_p_embed

    def forward_emb(self, g):
        n_embed, n_p_embed = self.gat(g)
        return n_embed, n_p_embed

# class GCNNet(nn.Module):
#     def __init__(self,
#                  n_layers, num_gat_layers,
#                  in_ch_list, base_ch_list, end_ch_list,
#                  checkpoint_layers, kernel_sizes,
#                  out_ch, padding_list, conv_strides, dropout,
#                  spatial_size, fv_dim, num_hiddens, node_embed_dim,
#                  norm_method='bn', act_method='relu'):
#         super(GCNNet, self).__init__()
#         self.fv_dim = fv_dim
#         self.dropout = dropout
#         self.n_layers = n_layers
#         self.in_ch_list = in_ch_list
#         self.base_ch_list = base_ch_list
#         self.checkpoint_layers = checkpoint_layers
#         self.conv_strides = conv_strides
#         self.spatial_size = spatial_size
#         self.kernel_sizes = kernel_sizes
#         self.num_hiddens = num_hiddens
#         self.node_embed_dim = node_embed_dim
#         self.end_ch_list = end_ch_list
#         assert (len(end_ch_list) == len(base_ch_list) == len(in_ch_list) == len(padding_list))
#         self.out_ch = out_ch
#         self.ds_modules = nn.ModuleList(
#             [
#                 ConvBlock5d([in_ch_list[n], base_ch_list[n]], [base_ch_list[n],
#                                                                end_ch_list[n]],
#                             checkpoint_layers[n], self.kernel_sizes[n], False, padding_list[n],
#                             conv_strides=conv_strides[n],
#                             norm_method=norm_method, act_method=act_method, dropout=dropout)
#                 for n in range(n_layers)
#             ]
#         )
#         self.bg = ConvBlock5d([in_ch_list[n_layers], base_ch_list[n_layers]],
#                               [base_ch_list[n_layers], end_ch_list[n_layers]],
#                               checkpoint_layers[n_layers], self.kernel_sizes[n_layers], False, padding_list[n_layers],
#                               dropout=dropout, norm_method=norm_method, act_method=act_method)
#
#         self.fc = nn.Sequential(
#             nn.Conv3d(end_ch_list[n_layers], end_ch_list[n_layers], kernel_size=self.spatial_size,
#                       padding=0, stride=1, bias=True),
#             nn.Dropout(dropout),
#             act_wrapper(act_method),
#             nn.Conv3d(end_ch_list[n_layers], fv_dim, kernel_size=1, padding=0, stride=1, bias=True),
#             act_wrapper(act_method)
#         )
#         self.out = nn.Conv3d(fv_dim, out_channels=out_ch, kernel_size=1, padding=0, bias=True)
#
#         self.gat = GCN(num_layers=num_gat_layers, in_dim=fv_dim,
#                        num_hiddens=self.num_hiddens, num_classes=node_embed_dim)
#         self.gnn_out = nn.Linear(node_embed_dim, out_ch)
#
#     def set_gcn_only(self):
#         set_trainable(self, False)
#         set_trainable(self.gat, True)
#         set_trainable(self.gnn_out, True)
#
#     def init(self, initializer):
#         initializer.initialize(self)
#         self.gat.reset_parameters()
#         nn.init.xavier_normal_(self.gnn_out.weight, gain=nn.init.calculate_gain('linear'))
#         nn.init.constant(self.gnn_out.bias, 0.0)
#
#     def extract_feature(self, x):
#         ds_feat_list = [x]
#         for idx, ds in enumerate(self.ds_modules):
#             ds_feat_list.append(ds(ds_feat_list[-1]))
#         xbg = self.bg(ds_feat_list[-1])
#         xbg = self.fc(xbg)
#         return xbg
#
#     def forward_without_gnn(self, x):
#         ds_feat_list = [x]
#         for idx, ds in enumerate(self.ds_modules):
#             if idx == 0:
#                 ds_feat_list.append(ds(ds_feat_list[-1]))
#             else:
#                 ds_feat_list.append(checkpoint(ds, ds_feat_list[-1]))
#         xbg = self.bg(ds_feat_list[-1])
#         xbg = self.fc(xbg)
#         return xbg, self.out(xbg)
#
#     def forward(self, g):
#         n_embed = self.gat(g)
#         n_out = self.gnn_out(n_embed)
#         return n_out, n_embed

#
# class Pool(nn.Module):
#
#     def __init__(self, k, in_dim, p):
#         super(Pool, self).__init__()
#         self.k = k
#         self.proj = nn.Linear(in_dim, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
#
#     def forward(self, graph, feature):
#         feature = self.drop(feature)
#         weights = self.proj(feature ).squeeze()
#         scores = self.sigmoid(weights)
#         select_idx, next_batch_num_nodes = topk(scores, self.k, get_batch_id(graph.batch_num_nodes()),
#                                           graph.batch_num_nodes())
#         graph = dgl.node_subgraph(graph, select_idx)
#
#         graph.set_batch_num_nodes(next_batch_num_nodes)
#
#         return graph, feature, select_idx
#
#
#
# class Unpool(nn.Module):
#
#     def __init__(self, *args):
#         super(Unpool, self).__init__()
#
#     def forward(self, graph, feat, select_idx):
#         fine_feat = torch.zeros((graph.num_nodes(), feat.size(-1)),
#                                 device=feat.device)
#         fine_feat[select_idx] = feat
#         return graph, fine_feat
#
#
# class GATPool(nn.Module):
#
#     def __init__(self, in_ch, out_ch, heads, activation,
#                  feat_drop,
#                  attn_drop,
#                  negative_slope,
#                  residual, k=0.5):
#         super(GATPool, self).__init__()
#         self.gat1 = GATConv(in_ch, out_ch, heads,
#                  feat_drop=feat_drop,
#                  attn_drop=attn_drop,
#                  negative_slope=negative_slope,
#                  residual=residual,
#                  activation=activation)
#         self.gat2 = GATConv(out_ch * heads, out_ch, heads,
#                  feat_drop=feat_drop,
#                  attn_drop=attn_drop,
#                  negative_slope=negative_slope,
#                  residual=residual,
#                  activation=activation)
#         self.pool = Pool(k, out_ch, feat_drop)
#
#     def forward(self, g, h):
#         h = self.gat1(g, h)
#         h = self.gat2(g, h)
#         sub_g, sub_h, sub_indices = self.pool(g, h)
#         sub_g.ndata['fvs'] = sub_h
#         sub_g.ndata['indices'] = sub_indices
#         return sub_g
#
#
# class GATUnPool(nn.Module):
#
#     def __init__(self, in_ch, out_ch, heads, activation,
#                  feat_drop,
#                  attn_drop,
#                  negative_slope,
#                  residual):
#         super(GATUnPool, self).__init__()
#         self.gat1 = GATConv(in_ch, out_ch, heads,
#                            feat_drop=feat_drop,
#                            attn_drop=attn_drop,
#                            negative_slope=negative_slope,
#                            residual=residual,
#                            activation=activation)
#         self.gat2 = GATConv(out_ch * heads, out_ch, heads,
#                            feat_drop=feat_drop,
#                            attn_drop=attn_drop,
#                            negative_slope=negative_slope,
#                            residual=residual,
#                            activation=activation)
#         self.unpool = Unpool(out_ch, feat_drop)
#
#     def forward(self, g, h, indices):
#         g, h = self.unpool(g, h, indices)
#         h = self.gat1(g, h )
#         h = self.gat2(g, h)
#         return h
#
# class GU(nn.Module):
#
#     def __init__(self, n_layers, ks, in_out_channels, heads, activation,
#                  feat_drop,
#                  attn_drop,
#                  negative_slope,
#                  residual):
#         super(GU, self).__init__()
#         self.ks = ks
#         self.n_layers = n_layers
#         self.in_out_tuple = in_out_channels
#         self.activation = activation
#         self.heads = heads
#         self.down_layers = nn.ModuleList()
#         self.up_layers = nn.ModuleList()
#
#         self.l_n = len(ks)
#         for i in range(n_layers):
#             self.down_layers.append(GATPool(in_out_channels[i][0], in_out_channels[i][1], heads[i],
#                 feat_drop, attn_drop, negative_slope, residual, self.activation, k=ks[i]))
#
#         self.bridge = GATConv(in_out_channels[n_layers][0], in_out_channels[n_layers][1], heads[n_layers],
#                 feat_drop, attn_drop, negative_slope, residual, self.activation)
#
#         for i in range(n_layers):
#             self.up_layers.append(GATUnPool(
#                 in_out_channels[n_layers + i + 1][0], in_out_channels[n_layers + i + 1][1], heads[n_layers + i + 1],
#                 feat_drop, attn_drop, negative_slope, residual, self.activation))
#
#
#     def reset_parameters(self):
#         for l in self.up_layers:
#             l.reset_parameters()
#
#         for l in self.down_layers:
#             l.reset_parameters()
#
#     def forward(self, g):
#         h = g.ndata['fvs']
#         g_cache = []
#         for down_layer in self.down_layers:
#             sub_g = down_layer(g, h).flatten(1)
#             g_cache.append((sub_g, g))
#         # output projection
#         logits = self.gat_layers[-1](g, h).mean(1)
#
#         adj_ms = []
#         indices_list = []
#         down_outs = []
#         hs = []
#         org_h = h
#         for i in range(self.l_n):
#             h = self.down_gcns[i](g, h)
#             adj_ms.append(g)
#             down_outs.append(h)
#             g, h, idx = self.pools[i](g, h)
#             indices_list.append(idx)
#         h = self.bottom_gcn(g, h)
#         for i in range(self.l_n):
#             up_idx = self.l_n - i - 1
#             g, idx = adj_ms[up_idx], indices_list[up_idx]
#             g, h = self.unpools[i](g, h, down_outs[up_idx], idx)
#             h = self.up_gcns[i](g, h)
#             h = h.add(down_outs[up_idx])
#             hs.append(h)
#         h = h.add(org_h)
#         hs.append(h)
#         return hs
#
# class GUNet(nn.Module):
#     def __init__(self,
#                  n_layers, num_gu_layers, num_heads,
#                  in_ch_list, base_ch_list, end_ch_list,
#                  checkpoint_layers, kernel_sizes,
#                  out_ch, padding_list, conv_strides, dropout, feat_drop, attn_drop, negative_slope,
#                  spatial_size, fv_dim, gu_in_out_channels, gu_ks, node_embed_dim,
#                  norm_method='bn', act_method='relu'):
#         super(GUNet, self).__init__()
#         self.fv_dim = fv_dim
#         self.dropout = dropout
#         self.n_layers = n_layers
#         self.in_ch_list = in_ch_list
#         self.base_ch_list = base_ch_list
#         self.checkpoint_layers = checkpoint_layers
#         self.conv_strides = conv_strides
#         self.spatial_size = spatial_size
#         self.kernel_sizes = kernel_sizes
#         self.gu_in_out_channels = gu_in_out_channels
#         self.gu_ks = gu_ks
#         self.num_gu_layers = num_gu_layers
#         self.node_embed_dim = node_embed_dim
#         self.end_ch_list = end_ch_list
#         assert (len(end_ch_list) == len(base_ch_list) == len(in_ch_list) == len(padding_list))
#         self.out_ch = out_ch
#         self.ds_modules = nn.ModuleList(
#             [
#                 ConvBlock5d([in_ch_list[n], base_ch_list[n]], [base_ch_list[n],
#                                                                end_ch_list[n]],
#                             checkpoint_layers[n], self.kernel_sizes[n], False, padding_list[n],
#                             conv_strides=conv_strides[n],
#                             norm_method=norm_method, act_method=act_method, dropout=dropout)
#                 for n in range(n_layers)
#             ]
#         )
#         self.bg = ConvBlock5d([in_ch_list[n_layers], base_ch_list[n_layers]],
#                               [base_ch_list[n_layers], end_ch_list[n_layers]],
#                               checkpoint_layers[n_layers], self.kernel_sizes[n_layers], False, padding_list[n_layers],
#                               dropout=dropout, norm_method=norm_method, act_method=act_method)
#
#         self.fc = nn.Sequential(
#             nn.Conv3d(end_ch_list[n_layers], end_ch_list[n_layers], kernel_size=self.spatial_size,
#                       padding=0, stride=1, bias=True),
#             nn.Dropout(dropout),
#             act_wrapper(act_method),
#             nn.Conv3d(end_ch_list[n_layers], fv_dim, kernel_size=1, padding=0, stride=1, bias=True),
#             act_wrapper(act_method)
#         )
#         self.out = nn.Conv3d(fv_dim, out_channels=out_ch, kernel_size=1, padding=0, bias=True)
#
#         self.gu = GU(n_layers=self.num_gu_layers, ks=gu_ks,
#                        in_out_channels=self.gu_in_out_channels, heads=num_heads,
#                        activation=F.elu, feat_drop=feat_drop, attn_drop=attn_drop,
#                        negative_slope=negative_slope, residual=True)
#         self.gnn_out = nn.Linear(node_embed_dim, out_ch)
#         self.gnn_lobe_out = nn.Linear(node_embed_dim, 6)
#         self.gnn_lung_out = nn.Linear(node_embed_dim, 3)
#
#     def set_gcn_only(self):
#         set_trainable(self, False)
#         set_trainable(self.gat, True)
#         set_trainable(self.gnn_out, True)
#         set_trainable(self.gnn_lobe_out, True)
#         set_trainable(self.gnn_lung_out, True)
#
#     def init(self, initializer):
#         initializer.initialize(self)
#         self.gat.reset_parameters()
#         nn.init.xavier_normal_(self.gnn_out.weight, gain=nn.init.calculate_gain('linear'))
#         nn.init.constant(self.gnn_out.bias, 0.0)
#
#     def extract_feature(self, x):
#         ds_feat_list = [x]
#         for idx, ds in enumerate(self.ds_modules):
#             ds_feat_list.append(ds(ds_feat_list[-1]))
#         xbg = self.bg(ds_feat_list[-1])
#         xbg = self.fc(xbg)
#         return xbg
#
#     def forward_without_gnn(self, x):
#         ds_feat_list = [x]
#         for idx, ds in enumerate(self.ds_modules):
#             if idx == 0:
#                 ds_feat_list.append(ds(ds_feat_list[-1]))
#             else:
#                 ds_feat_list.append(checkpoint(ds, ds_feat_list[-1]))
#         xbg = self.bg(ds_feat_list[-1])
#         xbg = self.fc(xbg)
#         return xbg, self.out(xbg)
#
#     def forward(self, g):
#         n_embed = self.gat(g)
#         n_out = self.gnn_out(n_embed)
#         return n_out, n_embed
#
#     def forward_all(self, g):
#         n_embed = self.gat(g)
#         n_out = self.gnn_out(n_embed)
#         n_lobe_out = self.gnn_lobe_out(n_embed)
#         n_lung_out = self.gnn_lung_out(n_embed)
#         return n_out, n_lobe_out, n_lung_out, n_embed