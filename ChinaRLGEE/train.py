import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from torch_geometric import datasets
from model import NetworkCORA as Network

from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.data import Data
from torch import optim

from ogb.graphproppred import PygGraphPropPredDataset

import utils

batch_size = 128


def Model(mat):
	if not torch.cuda.is_available():
		logging.info('no gpu device available')
		sys.exit(1)

	args, unknown = utils.parse_args()
	'''
	data_path = '/home/ustbai/tangyuxin/dataset/ZINC'
	# number of node attributes for ZINC dataset
	n_tags = 28
	num_edge_features = 4

	train_dset = datasets.ZINC(data_path, subset=True, split='train')
	train_data = []
	for j in range(len(train_dset)):
		train_data.append(
			Data(x=train_dset[j].x, edge_index=train_dset[j].edge_index, edge_attr=train_dset[j].edge_attr,
				 y=train_dset[j].y, in_degree=degree(train_dset[j].edge_index[0], train_dset[j].x.shape[0]).int()))

	input_size = n_tags
	train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

	val_dset = datasets.ZINC(data_path, subset=True, split='val')
	val_data = []
	for j in range(len(val_dset)):
		val_data.append(
			Data(x=val_dset[j].x, edge_index=val_dset[j].edge_index, edge_attr=val_dset[j].edge_attr, y=val_dset[j].y,
				 in_degree=degree(val_dset[j].edge_index[0], val_dset[j].x.shape[0]).int()))

	val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
	'''
	'''
    in_degree = []
    for i in range(len(train_dset)):
        in_degree.append(degree(train_dset[i].edge_index[0], train_dset[i].x.shape[0]).int().cuda())
    out_degree = in_degree
    '''
	dataset = PygGraphPropPredDataset(name = "ogbg-molesol", root=args.root) 
	'''
	np.random.seed(args.seed)
	torch.cuda.set_device(args.gpu)
	cudnn.benchmark = True
	torch.manual_seed(args.seed)
	cudnn.enabled = True
	torch.cuda.manual_seed(args.seed)
	'''
	genotype = eval("genotypes.%s" % args.arch)
	split_idx = dataset.get_idx_split()
	train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    
 
	max_degree = 0
    
    
	for i in range(len(dataset)):
		# data = dataset[i].to_dict
		# dataset[i] = Data(degree(dataset[i].edge_index[0], dataset[i].num_nodes).int())
		dataset[i].degree = degree(dataset[i].edge_index[0], dataset[i].num_nodes).int()
		if max_degree < max(degree(dataset[i].edge_index[0], dataset[i].num_nodes).int()):
			max_degree = max(degree(dataset[i].edge_index[0], dataset[i].num_nodes).int())
   
	train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
	valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
	test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
	num_class = 1
	input_dim = 9
 
	'''
	mat = []
	mat.append(np.matrix([[0, 0, 0, 1, 0, 0, 0, 0],
						  [0, 0, 0, 0, 0, 1, 0, 0],
						  [0, 0, 0, 0, 0, 0, 0, 4],
						  [0, 0, 0, 0, 0, 0, 0, 4],
						  [0, 0, 0, 0, 0, 0, 0, 4],
						  [0, 0, 0, 0, 0, 0, 0, 4],
						  [0, 0, 0, 0, 0, 0, 0, 0],
						  [0, 0, 0, 0, 0, 0, 0, 0]]))
	'''

	AllNetwork = []
	All_best_test_loss = []
	flag = 0
	for i in range(len(mat)):
		# AllNetwork.append(Network(mat[i], args, input_dim, genotype, num_class).cuda())
		model = Network(mat[i].astype(np.int_), args, input_dim, genotype, num_class, max_degree).cuda()
		# model = SuperNetwork(mat, args, input_dim)
		# model = model.cuda()

		# logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

		criterion = nn.MSELoss()
		optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

		best_valid_loss = 1000
		best_test_loss = 1000
		for epoch in range(args.epochs):
			train_loss = train(train_loader, model, criterion, optimizer)
			# valid_loss = infer(valid_loader, model, criterion)
			scheduler.step()
      
			'''
			for step, batch in enumerate(valid_loader):
      	batch = batch.cuda()
      	train_loss, train_logits = train(batch, model, criterion, optimizer)
                
			model.eval()
			for step, batch in enumerate(test_loader):
				batch = batch.cuda()
                
				valid_loss, valid_logits = infer(batch, model, criterion)
			scheduler.step()
			'''
			test_loss = infer(test_loader, model, criterion)
			if best_test_loss > test_loss:
				flag = 0
				# test_loss = infer(test_loader, model, criterion)
				best_test_loss = test_loss
				'''
				for step, batch in enumerate(test_loader):
					batch = batch.cuda()
                    
					test_loss, test_logits = infer(batch, model, criterion)
					best_test_loss = test_loss
				'''
			else:
				flag += 1
				if flag == 20:
					break
		print('best_test_loss %f', best_test_loss)
      
		All_best_test_loss.append(best_test_loss)
		torch.cuda.empty_cache()
	return All_best_test_loss

def train(loader, model, criterion, optimizer):
	model.train()
	running_loss = []
	for i, data in enumerate(loader):
		x = data.x.cuda()
		in_degree = degree(data.edge_index[0], data.num_nodes).int().cuda()
		out_degree = in_degree
		target = data.y.cuda(non_blocking=True)

		pointer = np.zeros(data.num_nodes).astype(np.int_)
		for j in range(len(data)):
			pointer[data.ptr[j]:data.ptr[j + 1]] = j

		optimizer.zero_grad()
		logits = model(x, data.edge_index.cuda(), data.edge_attr.cuda(), in_degree, out_degree, data.batch.cuda())

		loss = torch.sqrt(criterion(logits, target))
		loss.backward()
		# nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
		optimizer.step()

		running_loss.append(loss.item())

	epoch_loss = np.mean(running_loss)
	# print('train_loss:', epoch_loss)
	return epoch_loss


def infer(loader, model, criterion):
	model.eval()
	running_loss = []
	for i, data in enumerate(loader):
		x = data.x.cuda()
		in_degree = degree(data.edge_index[0], data.num_nodes).int().cuda()
		out_degree = in_degree
		target = data.y.cuda(non_blocking=True)

		pointer = np.zeros(data.num_nodes).astype(np.int_)
		for j in range(len(data)):
			pointer[data.ptr[j]:data.ptr[j + 1]] = j

		logits = model(x, data.edge_index.cuda(), data.edge_attr.cuda(), in_degree, out_degree, data.batch.cuda())
		loss = torch.sqrt(criterion(logits, target))
		running_loss.append(loss.item())

	epoch_loss = np.mean(running_loss)
	# print('val_loss:', epoch_loss)
	return epoch_loss


if __name__ == '__main__':
	Model([])