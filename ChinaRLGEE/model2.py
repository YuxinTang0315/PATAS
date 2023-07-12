import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import drop_path
import numpy as np
from torch_geometric.nn import GATConv, GINConv, GINEConv, SAGEConv
import torch_geometric.nn as gnn
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from graphpreprocessing2 import GRAPHPREPROCESS2 as GRAPHPREPROCESS


class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = edge_attr

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class graph_function(nn.Module):
	def __init__(self, op, hidden_dim):
		super(graph_function, self).__init__()
		if op == 'GraphSage':
			self.gcn = SAGEConv(hidden_dim, hidden_dim)
		elif op == 'GCN':
			self.gcn = GCNConv(hidden_dim)
		elif op=='GINE':
			self.gcn = GINEConv(nn.Linear(hidden_dim, hidden_dim), edge_dim=hidden_dim)
        
		self.norm = nn.BatchNorm1d(hidden_dim)
		self.relu = nn.ReLU()
		self.op = op

	def forward(self, x, edge_index, edge_attr):
		if self.op == 'GCN' or self.op == 'GINE':
			return self.norm(self.gcn(self.relu(x), edge_index, edge_attr))
		else:
			return self.norm(self.gcn(self.relu(x), edge_index))



def operation(op, hidden_dim):
	if op == 'GraphSage' or op == 'GCN' or op == 'GINE':
		return graph_function(op, hidden_dim)

	elif op == 'Transformer':
		return nn.TransformerEncoderLayer(d_model=hidden_dim,
										  nhead=8,
										  dim_feedforward=hidden_dim * 2,
										  norm_first=True)
	elif op == 'Big_Transformer':
		return nn.TransformerEncoderLayer(d_model=hidden_dim * 2,
														nhead=8,
														dim_feedforward=hidden_dim * 4,
														norm_first=True)


class Cell(nn.Module):
	def __init__(self, c_op, row, col, hidden_dim, hidden_dim_prev, hidden_dim_prev_prev, mid_node, dropout, in_degree):
		super(Cell, self).__init__()

		self.preprocess0 = nn.Sequential(
			nn.Linear(hidden_dim_prev_prev, hidden_dim),
			nn.BatchNorm1d(hidden_dim)
		)
		self.preprocess1 = nn.Sequential(
			nn.Linear(hidden_dim_prev, hidden_dim),
			nn.BatchNorm1d(hidden_dim)
		)
		self.b_linear = nn.Linear(hidden_dim, hidden_dim * 2)
		self.b_linear2 = nn.Linear(hidden_dim * 2, hidden_dim)

		if 'pe_Transformer' in c_op:
			self.in_degree_encoder = nn.Embedding(in_degree+1, hidden_dim)
			self.out_degree_encoder = nn.Embedding(in_degree+1, hidden_dim)

		self.dropout = torch.nn.Dropout(dropout)
		self.ops = nn.ModuleList()
		self.row = row
		self.col = col
		self.mid_node = mid_node
		self.name = c_op
		for op in c_op:
			self.ops += [operation(op, hidden_dim)]

	def forward(self, s0, s1, edge_index, edge_attr, in_degree, out_degree, mat, batch, max_node):
		# s0 = self.preprocess0(s0)
		# s1 = self.preprocess1(s1)
		node_embedding = []
		node_embedding.append(self.preprocess0(s0))
		node_embedding.append(self.preprocess1(s1))
		t = 0
		for i in range(self.mid_node):
			states = []
			list_col = self.col[self.col == 2 + i]
			list_row = self.row[self.col == 2 + i]
			for j in range(len(list_col)):
				if self.name[t] == 'GraphSage' or self.name[t] == 'GCN' or self.name[t] == 'GINE':
					states.append(self.dropout(self.ops[t](node_embedding[list_row[j]], edge_index, edge_attr)))
				else:
					embedding2 = node_embedding[list_row[j]]
					if self.name[t] == 'Big_Transformer':
						embedding2 = self.b_linear(embedding2)
					num_nodes = torch.unique(batch,return_counts=True)[1]
					embedding = torch.zeros([max_node, len(torch.unique(batch)), embedding2.shape[-1]]).cuda()                
					flag = 0                      
					mask = torch.zeros([len(torch.unique(batch)), max_node])     
					for k in range(len(torch.unique(batch))):
						embedding[ : num_nodes[k] - 1, k, :] = embedding2[flag : flag + num_nodes[k] - 1, :]
						mask[k, num_nodes[k]:] = 1
						flag += num_nodes[k]
					mask = torch.Tensor.bool(mask).cuda()
					embedding = self.dropout(self.ops[t](embedding, src_key_padding_mask=mask))
					if self.name[t] == 'Big_Transformer':
						embedding = self.b_linear2(embedding)
					embedding3 = torch.zeros(node_embedding[list_row[j]].shape).cuda()
					flag = 0 
					for k in range(len(torch.unique(batch))):
						embedding3[flag : flag + num_nodes[k] - 1, :] = embedding[ : num_nodes[k] - 1, k, :]
						flag += num_nodes[k]
					states.append(embedding3)                           
				t += 1
			node_embedding.append(sum(states))
		return torch.cat([node_embedding[i] for i in range(2, len(node_embedding))], dim=1)


class NetworkCORA(nn.Module):
	def __init__(self, mat, args, input_dim, genotype, num_class, in_degree):
		super(NetworkCORA, self).__init__()
		self.layers = args.layers
		self.mat = mat
		self.edge_attr_linear = nn.Linear(1, args.hidden_dim)

		# self.num_child = len(mat)

		hidden_dim_prev_prev, hidden_dim_prev, hidden_dim = args.hidden_dim * args.mid_node, args.hidden_dim * args.mid_node, args.hidden_dim
		self.children = nn.ModuleList()
		self.stem = nn.Sequential(
			AtomEncoder(hidden_dim_prev),
			nn.BatchNorm1d(hidden_dim_prev)
		)
   
		# self.GRAPHPREPROCESS = GRAPHPREPROCESS(hidden_size=384)
		self.bond_encoder = BondEncoder(hidden_dim)
		self.pool = global_mean_pool

		op_names, indices = zip(*genotype)
		c_op = []
		c_id = []
		# for i in range(len(mat)):
		self.cells = nn.ModuleList()
		col, row = np.nonzero(mat.T)

		c = 1
		while 1:
			if c >= len(col):
				break
			if col[c - 1] == col[c] and row[c] > 1:
				row = np.delete(row, [c])
				col = np.delete(col, [c])
			c += 1

		for r, c in zip(row, col):
			c_op.append(op_names[mat[r, c] - 1])
			c_id.append(indices[mat[r, c] - 1])
		
		row[row == 3] = 2
		row[row == 4] = 3
		row[row == 5] = 3
		row[row == 6] = 4
		row[row == 7] = 4
		col[col == 3] = 2
		col[col == 4] = 3
		col[col == 5] = 3
		col[col == 6] = 4
		col[col == 7] = 4
		for k in range(self.layers):
			cell = Cell(c_op, row, col, hidden_dim, hidden_dim_prev, hidden_dim_prev_prev, args.mid_node, args.dropout, in_degree)
			self.cells += [cell]

		# self.global_pooling = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Linear(hidden_dim_prev, num_class)

	def forward(self, x, edge_index, edge_attr, in_degree, out_degree, batch, max_node):
		# s0 = s1 = self.stem(x)
		s0 = s1 = x.to(torch.float32)
		# print(s0.shape)
		# print(edge_index.shape)
		# print(edge_attr.shape)
		# edge_attr = self.bond_encoder(edge_attr)
		edge_attr = self.edge_attr_linear(edge_attr)
		for i, cell in enumerate(self.cells):
			s0, s1 = s1, cell(s0, s1, edge_index, edge_attr, in_degree, out_degree, self.mat, batch, max_node)
		h_graph = self.pool(s1, batch)
		logits = self.classifier(h_graph)

		return logits