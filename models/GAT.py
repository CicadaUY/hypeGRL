import torch
import torch.nn.functional as F
from torch.nn import ReLU
from torch_geometric.nn import GATv2Conv, Sequential, TransformerConv


class GATv2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, num_heads=1):
        super(GATv2, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.gats = torch.nn.ModuleList()

        self.gats.append(
            Sequential(
                "x, edge_index",
                [
                    (GATv2Conv(input_dim, hidden_dim, heads=num_heads), "x, edge_index -> x"),
                    torch.nn.BatchNorm1d(num_features=hidden_dim),
                    ReLU(inplace=True),
                ],
            )
        )
        for _ in range(self.num_layers - 2):
            self.gats.append(
                Sequential(
                    "x, edge_index",
                    [
                        (GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads), "x, edge_index -> x"),
                        torch.nn.BatchNorm1d(num_features=hidden_dim),
                        ReLU(inplace=True),
                    ],
                )
            )

        self.gats.append(GATv2Conv(hidden_dim * num_heads, output_dim, heads=num_heads))
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, edge_index):

        for i in range(self.num_layers - 1):
            x = self.gats[i](x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gats[self.num_layers - 1](x, edge_index)
        out = self.softmax(x)

        return out


class GraphTransformer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout, num_heads=1):
        super(GraphTransformer, self).__init__()

        self.n_layers = n_layers
        self.dropout = dropout
        self.transformer = torch.nn.ModuleList()

        self.transformer.append(
            Sequential(
                "x, edge_index",
                [
                    (TransformerConv(input_dim, hidden_dim, heads=num_heads, bias=True), "x, edge_index -> x"),
                    torch.nn.BatchNorm1d(num_features=hidden_dim * num_heads),
                    ReLU(inplace=True),
                ],
            )
        )

        for _ in range(self.n_layers - 2):
            self.transformer.append(
                Sequential(
                    "x, edge_index",
                    [
                        (TransformerConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, bias=True), "x, edge_index -> x"),
                        torch.nn.BatchNorm1d(num_features=hidden_dim * num_heads),
                        ReLU(inplace=True),
                    ],
                )
            )

        self.transformer.append(TransformerConv(hidden_dim * num_heads, output_dim, heads=num_heads, bias=True))
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, edge_index):

        for i in range(self.n_layers - 1):
            x = self.transformer[i](x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.transformer[self.n_layers - 1](x, edge_index)

        return x
