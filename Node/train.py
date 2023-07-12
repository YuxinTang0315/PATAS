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
from model import NetworkCORA as Network

from data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

import utils

def Model(mat):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    
    args, unknown = utils.parse_args()
    
    dataset = Dataset(root=args.root, dataset=args.dataset)
    # print(dataset.data)
    # print(dataset.data.edge_index)
    # print(dataset.data.y)
    # print(dataset.data.train_mask)
    # dataset.data = utils.create_masks(dataset.data)
    
    in_degree = degree(dataset.data.edge_index[0], dataset.data.x.shape[0]).int().cuda()
    out_degree = in_degree
    num_class = len(np.unique(dataset.data.y))
    loader = DataLoader(dataset=dataset)
    
    input_dim = dataset.data.x.shape[-1]
    
    utils.create_masks(dataset.data)
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
    mat.append(np.matrix([[0,0,2,1,3,2,4,4],
                          [0,0,1,2,4,3,2,2],
                          [0,0,0,0,1,2,3,1],
                          [0,0,0,0,1,2,3,1],
                          [0,0,0,0,0,0,2,1],
                          [0,0,0,0,0,0,2,1],
                          [0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0]]))
    '''
    
    
    # AllNetwork = []
    All_best_test_acc = []
    for i in range(len(mat)):
        # AllNetwork.append(Network(mat[i], args, input_dim, genotype, num_class).cuda())
        model = Network(mat[i].astype(np.int_), args, input_dim, genotype, num_class, in_degree).cuda()
        # model = SuperNetwork(mat, args, input_dim)
        # model = model.cuda()

        # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
        
        best_valid_acc = 0
        best_train_acc = 0
        best_test_acc = 0
        mask = 0
        
        for epoch in range(args.epochs):
            # logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])

            train_loss, train_acc = train(dataset.data, model, criterion, optimizer, in_degree, out_degree, mask)
            # logging.info('train_acc %f', train_acc)
            scheduler.step()
            
            valid_loss, valid_acc, test_loss, test_acc = infer(dataset.data, model, criterion, in_degree, out_degree, mask)
            if best_valid_acc < valid_acc:
                best_train_acc = train_acc
                best_valid_acc = valid_acc
                best_test_acc = test_acc
        print('best_valid_acc %f', best_valid_acc)
        print('best_test_acc %f', best_test_acc)
        All_best_test_acc.append(best_test_acc)
        torch.cuda.empty_cache()
    return All_best_test_acc
    # return All_best_vaild_acc

    # utils.save(model, os.path.join(args.save, 'weights.pt'))
            
def train(data, model, criterion, optimizer, in_degree, out_degree, mask):
    model.train()
    x = data.x.cuda()
    target = data.y[data.train_mask[mask]].cuda(non_blocking=True)

    optimizer.zero_grad()
    logits = model(x, data.edge_index.cuda(), in_degree, out_degree)
    loss = criterion(logits[data.train_mask[mask]], target)
    loss.backward()
    # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()
    
    train_preds = torch.argmax(logits[data.train_mask[mask]], dim=1)
    train_acc = (torch.sum(train_preds == target).float() /
                       target.shape[0]).detach().cpu().numpy() * 100

    return loss, train_acc


def infer(data, model, criterion, in_degree, out_degree, mask):
    model.eval()
    x = data.x.cuda()
    target_val = data.y[data.val_mask[mask]].cuda(non_blocking=True)
    target_test = data.y[data.test_mask[mask]].cuda(non_blocking=True)

    logits = model(x, data.edge_index.cuda(), in_degree, out_degree)
    loss_val = criterion(logits[data.val_mask[mask]], target_val)
    loss_test = criterion(logits[data.test_mask[mask]], target_test)
    
    val_preds = torch.argmax(logits[data.val_mask[mask]], dim=1)
    val_acc = (torch.sum(val_preds == target_val).float() /
                       target_val.shape[0]).detach().cpu().numpy() * 100
                       
    test_preds = torch.argmax(logits[data.test_mask[mask]], dim=1)
    test_acc = (torch.sum(test_preds == target_test).float() /
                       target_test.shape[0]).detach().cpu().numpy() * 100

    return loss_val, val_acc, loss_test, test_acc


if __name__ == '__main__':
    Model([]) 

