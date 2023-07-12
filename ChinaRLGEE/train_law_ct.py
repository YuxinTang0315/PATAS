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
import torch.nn.functional as F

from torch.autograd import Variable
from model_law_ct import NetworkCORA as Network

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
from graphpreprocessing2 import GRAPHPREPROCESS2 as GRAPHPREPROCESS

import utils

def Model(mat):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    
    args, unknown = utils.parse_args()
    # dataset = Dataset(root=args.root, dataset=args.dataset)
    # dataset = MoleculeNet(root=args.root, name='HIV')
    
    _2013=np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2013.npy',allow_pickle=True)
    _2014=np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2014.npy',allow_pickle=True)
    _2015=np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2015.npy',allow_pickle=True)
    _2016=np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2016.npy',allow_pickle=True)
    _2017=np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2017.npy',allow_pickle=True)
    _2018=np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2018.npy',allow_pickle=True)
    
    _2013_=torch.from_numpy(np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2013_quota.npy',allow_pickle=True))
    _2014_=torch.from_numpy(np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2014_quota.npy',allow_pickle=True))
    _2015_=torch.from_numpy(np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2015_quota.npy',allow_pickle=True))
    _2016_=torch.from_numpy(np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2016_quota.npy',allow_pickle=True))
    _2017_=torch.from_numpy(np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2017_quota.npy',allow_pickle=True))
    _2018_=torch.from_numpy(np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2018_quota.npy',allow_pickle=True))
    
    _2013_y=torch.from_numpy(np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2013_y.npy',allow_pickle=True))
    _2014_y=torch.from_numpy(np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2014_y.npy',allow_pickle=True))
    _2015_y=torch.from_numpy(np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2015_y.npy',allow_pickle=True))
    _2016_y=torch.from_numpy(np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2016_y.npy',allow_pickle=True))
    _2017_y=torch.from_numpy(np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2017_y.npy',allow_pickle=True))
    _2018_y=torch.from_numpy(np.load('/home/ustbai/tangyuxin/self-supervised_learning/on_Graphs/Evolutionary_transformer_cls_graph/2018_y.npy',allow_pickle=True))
    
    __2013 = np.concatenate((_2013[0][0][np.newaxis, :], _2013[0][1][np.newaxis, :], _2013[0][2][np.newaxis, :]), axis=0)[np.newaxis, :]
    __2014 = np.concatenate((_2014[0][0][np.newaxis, :], _2014[0][1][np.newaxis, :], _2014[0][2][np.newaxis, :]), axis=0)[np.newaxis, :]
    __2015 = np.concatenate((_2015[0][0][np.newaxis, :], _2015[0][1][np.newaxis, :], _2015[0][2][np.newaxis, :]), axis=0)[np.newaxis, :]
    __2016 = np.concatenate((_2016[0][0][np.newaxis, :], _2016[0][1][np.newaxis, :], _2016[0][2][np.newaxis, :]), axis=0)[np.newaxis, :]
    __2017 = np.concatenate((_2017[0][0][np.newaxis, :], _2017[0][1][np.newaxis, :], _2017[0][2][np.newaxis, :]), axis=0)[np.newaxis, :]
    __2018 = np.concatenate((_2018[0][0][np.newaxis, :], _2018[0][1][np.newaxis, :], _2018[0][2][np.newaxis, :]), axis=0)[np.newaxis, :]
    for i in range(1, 53):
        __2013 = np.concatenate((__2013, np.concatenate((_2013[0][0][np.newaxis, :], _2013[0][1][np.newaxis, :], _2013[0][2][np.newaxis, :]), axis=0)[np.newaxis, :]), axis=0)
    for i in range(1, 100):
        __2014 = np.concatenate((__2014, np.concatenate((_2014[0][0][np.newaxis, :], _2014[0][1][np.newaxis, :], _2014[0][2][np.newaxis, :]), axis=0)[np.newaxis, :]), axis=0)
        __2015 = np.concatenate((__2015, np.concatenate((_2015[0][0][np.newaxis, :], _2015[0][1][np.newaxis, :], _2015[0][2][np.newaxis, :]), axis=0)[np.newaxis, :]), axis=0)
        __2016 = np.concatenate((__2016, np.concatenate((_2016[0][0][np.newaxis, :], _2016[0][1][np.newaxis, :], _2016[0][2][np.newaxis, :]), axis=0)[np.newaxis, :]), axis=0)
        __2017 = np.concatenate((__2017, np.concatenate((_2017[0][0][np.newaxis, :], _2017[0][1][np.newaxis, :], _2017[0][2][np.newaxis, :]), axis=0)[np.newaxis, :]), axis=0)
        __2018 = np.concatenate((__2018, np.concatenate((_2018[0][0][np.newaxis, :], _2018[0][1][np.newaxis, :], _2018[0][2][np.newaxis, :]), axis=0)[np.newaxis, :]), axis=0)
        
    train_graph_feats = torch.from_numpy(np.concatenate((__2013, __2014, __2015, __2016), axis=0)).cuda()
    train_graph_feats = F.normalize(train_graph_feats, dim=0).cuda()
    train_graph_feats = torch.cat((train_graph_feats[:, 0, :], train_graph_feats[:, 1, :], train_graph_feats[:, 2, :]), dim=1).cuda()
    train_graph_y = torch.from_numpy(np.concatenate((_2013_y, _2014_y, _2015_y, _2016_y), axis=0)).cuda()
    train_graph_y = F.normalize(train_graph_y, dim=0)
    
    valid_graph_feats = torch.from_numpy(__2017).cuda()
    valid_graph_feats = F.normalize(valid_graph_feats, dim=0).cuda()
    valid_graph_feats = torch.cat((valid_graph_feats[:, 0, :], valid_graph_feats[:, 1, :], valid_graph_feats[:, 2, :]), dim=1).cuda()
    valid_graph_y = _2017_y.cuda()
    valid_graph_y = F.normalize(valid_graph_y, dim=0)
    
    test_graph_feats = torch.from_numpy(__2018).cuda()
    test_graph_feats = F.normalize(test_graph_feats, dim=0).cuda()
    test_graph_feats = torch.cat((test_graph_feats[:, 0, :], test_graph_feats[:, 1, :], test_graph_feats[:, 2, :]), dim=1).cuda()
    test_graph_y = _2018_y.cuda()
    test_graph_y = F.normalize(test_graph_y, dim=0)
    
    train_scores = torch.cat((_2013_, _2014_, _2015_, _2016_), dim=0).cuda()
    valid_scores = _2017_.cuda()
    test_scores = _2018_.cuda()
    
    max_degree = 2
    max_node = 353
    
    num_class = 1
    input_dim = 128

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
    
    
    
    mat = []
    mat.append(np.matrix([[0,0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,0,1],
                          [0,0,0,0,0,2,0,1],
                          [0,0,0,0,0,2,0,1],
                          [0,0,0,0,0,0,0,1],
                          [0,0,0,0,0,0,0,1],
                          [0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0]]))
    
    
    
    AllNetwork = []
    All_best_test_loss = []
    
    # Rule of law government
    for i in range(len(mat)):
        flag = 0
        # AllNetwork.append(Network(mat[i], args, input_dim, genotype, num_class).cuda())
        model = Network(mat[i].astype(np.int_), args, input_dim, genotype, num_class, max_degree).cuda()
        g_process = GRAPHPREPROCESS(hidden_size=384).cuda()
        # model = SuperNetwork(mat, args, input_dim)
        # model = model.cuda()

        # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer2 = torch.optim.AdamW(g_process.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
        
        y_list = []
        valid_list = []
        label_list = []
        best_valid_loss = 100000000
        best_test_loss = 100000000
        for epoch in range(args.epochs):
            model.train()
            for i in range(1, 11):
                x, batch, edge_attr, graph_predict = g_process(train_graph_feats, train_scores)
                
                x2 = x[32*(i-1):32*i]
                x2 = x2.reshape(480, 384)
                
                edge_attr2 = edge_attr[32*(i-1):32*i]
                in_degree = torch.ones(x2.shape[0]).int().cuda()
                out_degree = in_degree
                batch2 = batch[15*32*(i-1):15*32*i]-32*(i-1)
                
                train_graph_y2 = train_graph_y[32*(i-1):32*i]
                
                edge_index = edge_attr2[0].to_sparse().indices().unsqueeze(0)
                edge_attr3 = edge_attr2[0].to_sparse().values().unsqueeze(1).unsqueeze(0)
                for j in range(1, edge_attr2.shape[0]):
                    edge_index = torch.cat((edge_index, edge_attr2[i].to_sparse().indices().unsqueeze(0) + 15*j), dim=0)
                    edge_attr3 = torch.cat((edge_attr3, edge_attr2[i].to_sparse().values().unsqueeze(1).unsqueeze(0)), dim=0)
                edge_index = edge_index.transpose(0, 1).reshape(2, -1)
                edge_attr3 = edge_attr3.reshape(-1, 1)
                
                train_loss, train_logits = train(x2, train_graph_y2, batch2, edge_index, edge_attr3, model, criterion, optimizer, optimizer2, in_degree, out_degree, max_node)
            x, batch, edge_attr, graph_predict = g_process(train_graph_feats, train_scores)
            x2 = x[320:]
            x2 = x2.reshape(495, 384)
            in_degree = torch.ones(x2.shape[0]).int().cuda()
            out_degree = in_degree
            edge_attr2 = edge_attr[320:]
            edge_index = edge_attr2[0].to_sparse().indices().unsqueeze(0)
            edge_attr3 = edge_attr2[0].to_sparse().values().unsqueeze(1).unsqueeze(0)
            for j in range(1, edge_attr2.shape[0]):
                edge_index = torch.cat((edge_index, edge_attr2[i].to_sparse().indices().unsqueeze(0) + 15*j), dim=0)
                edge_attr3 = torch.cat((edge_attr3, edge_attr2[i].to_sparse().values().unsqueeze(1).unsqueeze(0)), dim=0)
            edge_index = edge_index.transpose(0, 1).reshape(2, -1)
            edge_attr3 = edge_attr3.reshape(-1, 1)
            
            train_loss, train_logits = train(x[320:].reshape(495, 384), train_graph_y[320:], batch[4800:]-320, edge_index, edge_attr3, model, criterion, optimizer, optimizer2, in_degree, out_degree, max_node)
            
            model.eval()
            g_process.eval()
            x, batch, edge_attr, graph_predict = g_process(valid_graph_feats, valid_scores)
            x = x.reshape(1500, 384)
            in_degree = torch.ones(x.shape[0]).int().cuda()
            out_degree = in_degree
            edge_index = edge_attr[0].to_sparse().indices().unsqueeze(0)
            edge_attr2 = edge_attr[0].to_sparse().values().unsqueeze(1).unsqueeze(0)
            for j in range(1, edge_attr.shape[0]):
                edge_index = torch.cat((edge_index, edge_attr[i].to_sparse().indices().unsqueeze(0) + 15*j), dim=0)
                edge_attr2 = torch.cat((edge_attr2, edge_attr[i].to_sparse().values().unsqueeze(1).unsqueeze(0)), dim=0)
            
            edge_index = edge_index.transpose(0, 1).reshape(2, -1)
            edge_attr2 = edge_attr2.reshape(-1, 1)
            
            valid_loss, valid_logits = infer(x, valid_graph_y, batch, edge_index, edge_attr2, model, criterion, in_degree, out_degree, max_node)
                
            # scheduler.step()
            
            if best_valid_loss > valid_loss:
                flag = 0
                x, batch, edge_attr, graph_predict = g_process(test_graph_feats, test_scores)
                x = x.reshape(1500, 384)
                in_degree = torch.ones(x.shape[0]).int().cuda()
                out_degree = in_degree
                edge_index = edge_attr[0].to_sparse().indices().unsqueeze(0)
                edge_attr2 = edge_attr[0].to_sparse().values().unsqueeze(1).unsqueeze(0)
                for j in range(1, edge_attr.shape[0]):
                    edge_index = torch.cat((edge_index, edge_attr[i].to_sparse().indices().unsqueeze(0) + 15*j), dim=0)
                    edge_attr2 = torch.cat((edge_attr2, edge_attr[i].to_sparse().values().unsqueeze(1).unsqueeze(0)), dim=0)
                
                edge_index = edge_index.transpose(0, 1).reshape(2, -1)
                edge_attr2 = edge_attr2.reshape(-1, 1)
                
                test_loss, test_logits = infer(x, test_graph_y, batch, edge_index, edge_attr2, model, criterion, in_degree, out_degree, max_node)
            
                best_test_loss = test_loss
            else:
                flag += 1
                if flag == 20:
                    break
        # print('best_valid_auc %f', best_valid_auc)
        print('best_test_loss %f', best_test_loss)
        All_best_test_loss.append(best_test_loss.item())
        torch.cuda.empty_cache()
    return All_best_test_loss
            
def train(x, train_graph_y, batch, edge_index, edge_attr, model, criterion, optimizer, optimizer2, in_degree, out_degree, max_node):
    target = train_graph_y

    optimizer.zero_grad()
    optimizer2.zero_grad()
    logits = model(x, edge_index, edge_attr, in_degree, out_degree, batch, max_node)
    loss = criterion(logits, target.to(torch.float32))
    loss.backward()
    # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()
    optimizer2.step()

    return loss, logits


def infer(x, valid_graph_y, batch, edge_index, edge_attr, model, criterion, in_degree, out_degree, max_node):
    target = valid_graph_y
    
    logits = model(x, edge_index, edge_attr, in_degree, out_degree, batch, max_node)
    loss = criterion(logits, target.to(torch.float32))

    return loss, logits


if __name__ == '__main__':
    Model([]) 

