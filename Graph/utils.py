from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikiCS
import torch.nn.functional as F

import os.path as osp
import os

import argparse

import scipy.sparse as sp
import numpy as np
# np.random.seed(0)

import torch
import torch.nn as nn
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
from datetime import datetime
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable


def currentTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--embedder", type=str, default="DBYOL")
    parser.add_argument("--dataset", type=str, default="sider", help="Name of the dataset. Supported names are: cora, citeseer")
    parser.add_argument('--checkpoint_dir', type=str, default = './model_checkpoints', help='directory to save checkpoint')
    parser.add_argument("--root", type=str, default="/home/ustbai/tangyuxin/dataset")
    parser.add_argument("--task", type=str, default="node", help="Downstream task. Supported tasks are: node, clustering, similarity")
    parser.add_argument("--pred_hid", type=int, default=2048, help="The number of hidden units of layer of the predictor. Default is 512")
    
    parser.add_argument("--bn", type=bool, default=True)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--step_size", type=int, default=600)
    parser.add_argument("--gamma", type=int, default=0.5)
    parser.add_argument("--batchsize", type=int, default=32)
    
    parser.add_argument("--alpha", type=float, default=0.2, help="The frequency of model evaluation")
    parser.add_argument("--mad", type=float, default=0.999, help="Moving Average Decay for Teacher Network")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")    
    parser.add_argument("--es", type=int, default=25, help="Early Stopping Criterion")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.9)
    
    parser.add_argument('--arch', type=str, default='Graph_Transformer', help='which architecture to use')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument("--layers", type=int, default=1, help="The number of cells")
    parser.add_argument("--mid_node", type=int, default=3)
    

    return parser.parse_known_args()


def decide_config(root, dataset):
    """
    Create a configuration to download datasets
    :param root: A path to a root directory where data will be stored
    :param dataset: The name of the dataset to be downloaded
    :return: A modified root dir, the name of the dataset class, and parameters associated to the class
    """
    dataset = dataset.lower()
    if dataset == 'cora' or dataset == 'citeseer' or dataset == "pubmed":
        root = osp.join(root, "pyg", "planetoid")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Planetoid, "src": "pyg"}
    elif dataset == "computers":
        dataset = "Computers"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Amazon, "src": "pyg"}
    elif dataset == "photo":
        dataset = "Photo"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Amazon, "src": "pyg"}
    elif dataset == "cs" :
        dataset = "CS"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Coauthor, "src": "pyg"}
    elif dataset == "physics":
        dataset = "Physics"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Coauthor, "src": "pyg"}
    elif dataset == "wikics":
        dataset = "WikiCS"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root},
                  "name": dataset, "class": WikiCS, "src": "pyg"}
    else:
        raise Exception(
            f"Unknown dataset name {dataset}, name has to be one of the following 'cora', 'citeseer', 'pubmed', 'photo', 'computers', 'cs', 'physics'")
    return params

def create_masks(data):
    """
    Splits data into training, validation, and test splits in a stratified manner if
    it is not already splitted. Each split is associated with a mask vector, which
    specifies the indices for that split. The data will be modified in-place
    :param data: Data object
    :return: The modified data
    """
    # data.train_mask = data.dev_mask = data.test_mask = None

    # Build train_mask and val_mask and test_mask for 20 times
    '''
    for i in range(20):
        labels = data.y.numpy()
        dev_size = int(labels.shape[0] * 0.1)
        test_size = int(labels.shape[0] * 0.8)

        perm = np.random.permutation(labels.shape[0])
        test_index = perm[:test_size]
        dev_index = perm[test_size:test_size + dev_size]

        data_index = np.arange(labels.shape[0])
        test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
        dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
        train_mask = ~(dev_mask + test_mask)
        test_mask = test_mask.reshape(1, -1)
        dev_mask = dev_mask.reshape(1, -1)
        train_mask = train_mask.reshape(1, -1)

        if 'train_mask' not in data:
            data.train_mask = train_mask
            data.val_mask = dev_mask
            data.test_mask = test_mask
        else:
            data.train_mask = torch.cat((data.train_mask, train_mask), dim=0)
            data.val_mask = torch.cat((data.val_mask, dev_mask), dim=0)
            data.test_mask = torch.cat((data.test_mask, test_mask), dim=0)
    '''
    labels = data.y.numpy()
    dev_size = int(labels.shape[0] * 0.2)
    test_size = int(labels.shape[0] * 0.2)

    perm = np.random.permutation(labels.shape[0])
    test_index = perm[:test_size]
    dev_index = perm[test_size:test_size + dev_size]

    data_index = np.arange(labels.shape[0])
    test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
    dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
    train_mask = ~(dev_mask + test_mask)
    test_mask = test_mask.reshape(1, -1)
    dev_mask = dev_mask.reshape(1, -1)
    train_mask = train_mask.reshape(1, -1)
    
    data.train_mask = train_mask
    data.val_mask = dev_mask
    data.test_mask = test_mask

    return data

def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        sub_dirs.pop(0)
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)



class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

