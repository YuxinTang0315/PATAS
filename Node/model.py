import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import drop_path
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
import torch_geometric.nn as gnn


class graph_function(nn.Module):
	def __init__(self, op, hidden_dim):
		super(graph_function, self).__init__()
		if op == 'GraphSage':
			self.gcn = SAGEConv(hidden_dim, hidden_dim)
		elif op == 'GIN':
			self.gcn = GINConv(nn.Linear(hidden_dim, hidden_dim))
		else:
			self.gcn = GCNConv(hidden_dim, hidden_dim)

		self.norm = nn.BatchNorm1d(hidden_dim)
		self.relu = nn.ReLU()

	def forward(self, x, edge_index):
		return self.norm(self.gcn(self.relu(x), edge_index=edge_index))


def operation(op, hidden_dim):
	if op == 'GraphSage' or op == 'GCN' or op == 'GIN':
		return graph_function(op, hidden_dim)

	elif op == 'Transformer' or op == 'pe_Transformer':
		return nn.TransformerEncoderLayer(d_model=hidden_dim,
										  nhead=8,
										  dim_feedforward=hidden_dim * 2,
										  norm_first=True)
	elif op == 'Big_Transformer':
		return nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
							 nn.TransformerEncoderLayer(d_model=hidden_dim * 2,
														nhead=8,
														dim_feedforward=hidden_dim * 4,
														norm_first=True),
							 nn.Linear(hidden_dim * 2, hidden_dim))


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

		if 'pe_Transformer' in c_op:
			self.in_degree_encoder = nn.Embedding(max(in_degree)+1, hidden_dim)
			self.out_degree_encoder = nn.Embedding(max(in_degree)+1, hidden_dim)

		self.dropout = torch.nn.Dropout(dropout)
		self.ops = nn.ModuleList()
		self.row = row
		self.col = col
		self.mid_node = mid_node
		self.name = c_op
		for op in c_op:
			self.ops += [operation(op, hidden_dim)]

	def forward(self, s0, s1, edge_index, in_degree, out_degree, mat):
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
				if self.name[t] == 'GraphSage' or self.name[t] == 'GIN' or self.name[t] == 'GCN':
					states.append(self.dropout(self.ops[t](node_embedding[list_row[j]], edge_index)))
				else:
					if self.name[t] == 'pe_Transformer':
						node_feature = (
								node_embedding[list_row[j]]
								+ self.in_degree_encoder(in_degree)
								+ self.out_degree_encoder(out_degree)
						)
						states.append(self.dropout(self.ops[t](node_feature.unsqueeze(1)).squeeze(1)))
					else:
						states.append(self.dropout(self.ops[t](node_embedding[list_row[j]].unsqueeze(1)).squeeze(1)))
				t += 1
			node_embedding.append(sum(states))
		return torch.cat([node_embedding[i] for i in range(2, len(node_embedding))], dim=1)


class NetworkCORA(nn.Module):
	def __init__(self, mat, args, input_dim, genotype, num_class, in_degree):
		super(NetworkCORA, self).__init__()
		self.layers = args.layers
		self.mat = mat

		# self.num_child = len(mat)

		hidden_dim_prev_prev, hidden_dim_prev, hidden_dim = args.hidden_dim * args.mid_node, args.hidden_dim * args.mid_node, args.hidden_dim
		self.children = nn.ModuleList()
		self.stem = nn.Sequential(
			nn.Linear(input_dim, hidden_dim_prev),
			nn.BatchNorm1d(hidden_dim_prev)
		)

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

	def forward(self, x, edge_index, in_degree, out_degree):
		s0 = s1 = self.stem(x)
		for i, cell in enumerate(self.cells):
			s0, s1 = s1, cell(s0, s1, edge_index, in_degree, out_degree, self.mat)
		logits = self.classifier(s1)
		return logits