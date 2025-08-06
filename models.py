import torch
import torch.nn as nn
import torch.nn.functional as F

# 图注意力层
class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.a = nn.Parameter(torch.FloatTensor(2 * output_dim, 1))
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, input, adj):
        Wh = torch.mm(input, self.weight)
        Wh1 = torch.matmul(Wh, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(Wh, self.a[self.output_dim:, :])
        e = Wh1 + Wh2.T
        e = self.leakyrelu(e)

        if adj.is_sparse:
            adj = adj.coalesce()
            indices = adj.indices()
            values = adj.values()
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        else:
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)  # (N, output_dim)
        return h_prime


# 图卷积层
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, bias=False):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.reset_parameters()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = self.dropout(input)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


# 节点聚合模型
class NodeAggregationModel(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, hidden_dim3, num_heads, dropout, alpha):
        super(NodeAggregationModel, self).__init__()
        self.attentions = nn.ModuleList(
            [GraphAttentionLayer(input_feat_dim, hidden_dim1, dropout=dropout, alpha=alpha) for _ in range(num_heads)]
        )
        self.gc1 = GraphConvolution(hidden_dim1 * num_heads, hidden_dim2, dropout)
        self.gc2 = GraphConvolution(hidden_dim2, hidden_dim3, dropout)
        self.relu = nn.ReLU()
        self.dropout = dropout

    def forward(self, input, adj):
        x = torch.cat([att(input, adj) for att in self.attentions], dim=-1)  # 拼接多头注意力的输出
        x = self.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc1(x, adj)
        x = self.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x, adj)

        # 返回节点特征表示和Softmax结果
        x_softmax = F.softmax(x, dim=1)
        return x, x_softmax