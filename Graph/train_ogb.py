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
from tqdm import tqdm

from torch.autograd import Variable
from model_ogb import NetworkCORA as Network

from data import Dataset
# from torch_geometric.loader import DataLoader
# from torch.utils.data import DataLoader
from torch_geometric.utils import degree
from torch_geometric.datasets import MoleculeNet

from ogb.graphproppred import GraphPropPredDataset

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from kill_node import process

import utils

def Model(mat):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    
    args, unknown = utils.parse_args()
    dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root=args.root) 
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    
    max_degree = 0
    max_node = 0
    
    for i in range(len(dataset)):
      # data = dataset[i].to_dict
      # dataset[i] = Data(degree(dataset[i].edge_index[0], dataset[i].num_nodes).int())
      dataset[i].degree = degree(dataset[i].edge_index[0], dataset[i].num_nodes).int()
      if max_degree < max(degree(dataset[i].edge_index[0], dataset[i].num_nodes).int()):
        max_degree = max(degree(dataset[i].edge_index[0], dataset[i].num_nodes).int())
      if max_node < dataset[i].num_nodes:
        max_node = dataset[i].num_nodes
    
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batchsize, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batchsize, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batchsize, shuffle=False)
    num_class = 1
    input_dim = 9
    
    # input_dim = dataset.data.x.shape[-1]
    '''
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    '''

    genotype = eval("genotypes.%s" % args.arch)
    '''
    mat = []
    mat.append(np.matrix([[0,0,1,0,3,0,4,4],
                          [0,0,1,2,0,3,2,0],
                          [0,0,0,0,0,5,0,1],
                          [0,0,0,0,0,5,0,1],
                          [0,0,0,0,0,0,2,0],
                          [0,0,0,0,0,0,2,0],
                          [0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0]]))
    '''
    
    
    AllNetwork = []
    All_best_test_auc = []
    for i in range(len(mat)):
        flag = 0
        # AllNetwork.append(Network(mat[i], args, input_dim, genotype, num_class).cuda())
        model = Network(mat[i].astype(np.int_), args, input_dim, genotype, num_class, max_degree).cuda()
        # model = SuperNetwork(mat, args, input_dim)
        # model = model.cuda()

        # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        criterion = torch.nn.BCEWithLogitsLoss()
        criterion = criterion.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
        
        y_list = []
        valid_list = []
        label_list = []
        best_valid_auc = 0
        best_test_auc = 0
        for epoch in range(args.epochs):
            for step, batch in enumerate(train_loader):
                batch = batch.cuda()
                in_degree = degree(batch.edge_index[0], batch.num_nodes).int().cuda()
                out_degree = in_degree
                
                # label_list += batch.y.tolist()
                train_loss, train_logits = train(batch, model, criterion, optimizer, in_degree, out_degree, max_node)
                # valid_list += train_logits.tolist()
            # auc_score = roc_auc_score(label_list, valid_list)
            # label_list.clear()
            # valid_list.clear()
            # print('train_auc:', auc_score)
            for step, batch in enumerate(valid_loader):
                batch = batch.cuda()
                in_degree = degree(batch.edge_index[0], batch.num_nodes).int().cuda()
                out_degree = in_degree
                
                label_list += batch.y.tolist()
                valid_loss, valid_logits = infer(batch, model, criterion, in_degree, out_degree, max_node)
                
                valid_list += valid_logits.tolist()
                
            scheduler.step()
            auc_score = roc_auc_score(label_list, valid_list)
            label_list.clear()
            valid_list.clear()
            # print('valid_auc:', auc_score)
            if best_valid_auc < auc_score:
                flag = 0
                for step, batch in enumerate(test_loader):
                    batch = batch.cuda()
                    in_degree = degree(batch.edge_index[0], batch.num_nodes).int().cuda()
                    out_degree = in_degree
                    
                    label_list += batch.y.tolist()
                    
                    test_loss, test_logits = infer(batch, model, criterion, in_degree, out_degree, max_node)
                    y_list += test_logits.tolist()
                best_valid_auc = auc_score
                best_test_auc = roc_auc_score(label_list, y_list)
                # print(best_test_auc)
                label_list.clear()
                y_list.clear()
            else:
                flag += 1
                if flag == 30:
                    break
        # print('best_valid_auc %f', best_valid_auc)
        print('best_test_auc %f', best_test_auc)
        All_best_test_auc.append(best_test_auc)
        torch.cuda.empty_cache()
    return All_best_test_auc
    # return All_best_vaild_acc

    # utils.save(model, os.path.join(args.save, 'weights.pt'))
            
def train(data, model, criterion, optimizer, in_degree, out_degree, max_node):
    model.train()
    x = data.x
    target = data.y

    optimizer.zero_grad()
    logits = model(x, data.edge_index, data.edge_attr, in_degree, out_degree, data.batch, max_node)
    loss = criterion(logits, target.to(torch.float32))
    loss.backward()
    # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    return loss, logits


def infer(data, model, criterion, in_degree, out_degree, max_node):
    model.eval()
    with torch.no_grad():
        x = data.x
        target = data.y
    
        logits = model(x, data.edge_index, data.edge_attr, in_degree, out_degree, data.batch, max_node)
        loss = criterion(logits, target.to(torch.float32))
        # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

    return loss, logits


if __name__ == '__main__':
    Model([]) 

