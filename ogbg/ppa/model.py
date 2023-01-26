import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl import function as fn
from dgl.ops.edge_softmax import edge_softmax
from utils.jumping_knowledge import JumpingKnowledge
# from torch.nn import Linear, Sequential, Tanh, PReLU, ReLU, Hardswish, GELU, LeakyReLU, BatchNorm1d

from utils.covariance import compute_covariance

class Net(torch.nn.Module):
    def __init__(self,
                 config,
                 num_class,
                 num_basis):
        super(Net, self).__init__()

        self.node_encoder = torch.nn.Embedding(1, config.hidden)

        self.convs = torch.nn.ModuleList()
        for i in range(config.layers):
            self.convs.append(Conv(hidden_size=config.hidden, dropout_rate=config.dropout, exp_n=config.exp_n))

        self.emb_jk = JumpingKnowledge('C')
        self.cov_jk = JumpingKnowledge('S')
        self.graph_pred_linear = torch.nn.Linear(config.hidden * config.layers, num_class)

        if config.pooling == 'S':
            self.pool = SumPooling()
        elif config.pooling == 'M':
            self.pool = AvgPooling()
        elif config.pooling == 'X':
            self.pool = MaxPooling()

        self.filter_encoder = nn.Sequential(
            nn.Linear(num_basis, config.hidden * config.exp_n),
            # nn.BatchNorm1d(config.hidden * config.exp_n),
            nn.GELU(),
            nn.Linear(config.hidden * config.exp_n, config.hidden * config.exp_n),
            # nn.BatchNorm1d(config.hidden * config.exp_n),
            nn.GELU()
        )
        self.dropout = config.dropout

    def forward(self, g, x, edge_attr, bases):
        x = self.node_encoder(x)
        bases = self.filter_encoder(bases)
        # bases = edge_softmax(g, bases)
        xs = []
        xtxs = []
        for conv in self.convs:
            x, xtx = conv(g, x, edge_attr, bases)
            xtxs = xtxs + [xtx]
            xs = xs + [x]
        x = self.emb_jk(xs)
        x = F.dropout(x, p=self.dropout, training=self.training)
        h_graph = self.pool(g, x)
        return self.graph_pred_linear(h_graph), self.cov_jk(xtxs)

    def __repr__(self):
        return self.__class__.__name__


class Conv(nn.Module):
    def __init__(self, hidden_size, dropout_rate, exp_n):
        super(Conv, self).__init__()
        self.pre_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * exp_n),
            # nn.BatchNorm1d(hidden_size * exp_n),
            nn.ReLU()
        )
        self.pre_ffn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * exp_n),
            # nn.BatchNorm1d(hidden_size * exp_n),
            nn.ReLU()
        )
        # self.preffn_dropout = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size * exp_n, hidden_size * exp_n),
            nn.ReLU(),
            nn.Linear(hidden_size * exp_n, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size)
        )
        # self.ffn_dropout = nn.Dropout(dropout_rate)
        self.edge_encoder = nn.Linear(7, hidden_size)

    def forward(self, graph, x_feat, edge_attr, bases):
        with graph.local_scope():
            graph.ndata['x'] = x_feat
            graph.edata['e'] = self.edge_encoder(edge_attr)
            graph.apply_edges(fn.u_add_e('x', 'e', 'pos_e'))
            graph.edata['v'] = self.pre_ffn(graph.edata['pos_e']) * bases
            graph.update_all(fn.copy_e('v', 'pre_aggr'), fn.sum('pre_aggr', 'aggr'))
            y = graph.ndata['aggr'] + self.pre_ffn2(x_feat)
            # y = self.preffn_dropout(y)
            xtx = compute_covariance(y)
            y = self.ffn(y)
            # y = self.ffn_dropout(y)
            return y, xtx
