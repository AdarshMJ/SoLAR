import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import SGConv
from torch.nn import Sequential as Seq, Linear, ReLU, Dropout, GELU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# class GCN(torch.nn.Module):
#     def __init__(self, num_features, num_classes, hidden_channels, num_layers):
#         super().__init__()
#         torch.manual_seed(12345)
#         self.convs = torch.nn.ModuleList()
#         self.convs.append(GCNConv(num_features, hidden_channels))
#         for _ in range(num_layers - 2):  # -2 because we add the first and last layers separately
#             self.convs.append(GCNConv(hidden_channels, hidden_channels))
#         self.convs.append(GCNConv(hidden_channels, num_classes))

#     def forward(self, x, edge_index, p):
#         for i in range(len(self.convs) - 1):
#             x = self.convs[i](x, edge_index)
#             x = x.relu()
#             x = F.dropout(x, p=p, training=self.training)
#         x = self.convs[-1](x, edge_index)
#         return x

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, use_bn=True):
        super().__init__()
        torch.manual_seed(12345)

        cached = False
        add_self_loops = True
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GATv2(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=16):
        torch.manual_seed(12345)
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=heads)

    def forward(self, x, edge_index):
        h = F.dropout(x, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, training=self.training)
        h = self.gat2(h, edge_index)
        return F.log_softmax(h, dim=1)

    def reset_parameters(self):
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()

class SimpleGCN(torch.nn.Module):
    def __init__(self, num_features,num_classes,hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.31, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def reset_parameters(self):
            self.conv1.reset_parameters()
            self.conv2.reset_parameters()

class SGC(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = SGConv(
            in_channels=num_features,
            out_channels=num_classes,
            K=2,
            cached=True,
        )

    def forward(self,x,edge_index,p=0.0):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)



# class FeedForwardModule(torch.nn.Module):
#     def __init__(self, dim, hidden_dim_multiplier, dropout, input_dim_multiplier=1, **kwargs):
#         super().__init__()
#         input_dim = int(dim * input_dim_multiplier)
#         hidden_dim = int(dim * hidden_dim_multiplier)
#         self.linear_1 = Linear(in_features=input_dim, out_features=hidden_dim)
#         self.dropout_1 = Dropout(p=dropout)
#         self.act = GELU()
#         self.linear_2 = Linear(in_features=hidden_dim, out_features=dim)
#         self.dropout_2 = Dropout(p=dropout)

#     def forward(self, x):
#         x = self.linear_1(x)
#         x = self.dropout_1(x)
#         x = self.act(x)
#         x = self.linear_2(x)
#         x = self.dropout_2(x)

#         return x


# class newGCN(MessagePassing):
#     def __init__(self, dim, hidden_dim_multiplier, dropout, **kwargs):
#         super().__init__(aggr='add')  # "Add" aggregation.
#         self.feed_forward_module = FeedForwardModule(dim=dim,
#                                                      hidden_dim_multiplier=hidden_dim_multiplier,
#                                                      dropout=dropout)

#     def forward(self, x, edge_index):
#         # Step 1: Add self-loops to the adjacency matrix.
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

#         # Step 2: Linearly transform node feature matrix.
#         x = self.feed_forward_module(x)

#         # Step 3: Compute normalization.
#         row, col = edge_index
#         deg = degree(row, x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         # Step 4: Propagate the embeddings to the next layer.
#         return self.propagate(edge_index, x=x, norm=norm)

#     def message(self, x_j, norm):
#         # Normalize node features.
#         return norm.view(-1, 1) * x_j