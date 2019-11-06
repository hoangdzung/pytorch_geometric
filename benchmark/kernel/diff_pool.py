from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, JumpingKnowledge


class Block(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, mode='cat'):
        super(Block, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, out_channels)
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin = Linear(hidden_channels + out_channels, out_channels)
        else:
            self.lin = Linear(out_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, adj, mask, add_loop))
        x2 = F.relu(self.conv2(x1, adj, mask, add_loop))
        return self.lin(self.jump([x1, x2]))


class DiffPool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.25):
        super(DiffPool, self).__init__()

        num_nodes = ceil(ratio * dataset[0].num_nodes)
        self.embed_block1 = Block(dataset.num_features, hidden, hidden)
        self.pool_block1 = Block(dataset.num_features, hidden, num_nodes)

        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        for i in range((num_layers // 2) - 1):
            num_nodes = ceil(ratio * num_nodes)
            self.embed_blocks.append(Block(hidden, hidden, hidden))
            self.pool_blocks.append(Block(hidden, hidden, num_nodes))

        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear((len(self.embed_blocks) + 1) * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.pool_block1.reset_parameters()
        for embed_block, pool_block in zip(self.embed_blocks,
                                           self.pool_blocks):
            embed_block.reset_parameters()
            pool_block.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def adj_loss(self, adj):
        n_cluster = adj.shape[1]
        diag_batch = torch.diagonal(adj,dim1=1,dim2=2)
        return torch.sum((diag_batch - torch.mean(diag_batch,-1,True))**2)

    def s_loss(self, s):
        s_batch = s.sum(1)
        return torch.sum((s_batch - torch.mean(s_batch,-1,True))**2)

    def forward(self, data, weight=[0,0,0,0]):
        x, adj, mask = data.x, data.adj, data.mask

        s = self.pool_block1(x, adj, mask, add_loop=True)
        x = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
        xs = [x.mean(dim=1)]

        link_losses = []
        entropy_losses = []
        adj_losses = []
        s_losses = []

        x, adj, l, e = dense_diff_pool(x, adj, s, mask)

        link_losses.append(l)
        entropy_losses.append(e)
        adj_losses.append(self.adj_loss(adj))
        s_losses.append(self.s_loss(s))

        for i, (embed_block, pool_block) in enumerate(
                zip(self.embed_blocks, self.pool_blocks)):
            s = pool_block(x, adj)
            x = F.relu(embed_block(x, adj))
            xs.append(x.mean(dim=1))
            if i < len(self.embed_blocks) - 1:
                x, adj, l, e = dense_diff_pool(x, adj, s)
                link_losses.append(l)
                entropy_losses.append(e)
                adj_losses.append(self.adj_loss(adj))
                s_losses.append(self.s_loss(s))

        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), sum(link_losses)*weight[0] + sum(entropy_losses)*weight[1] + sum(adj_losses)*weight[2]+sum(s_losses)*weight[3]

    def __repr__(self):
        return self.__class__.__name__
