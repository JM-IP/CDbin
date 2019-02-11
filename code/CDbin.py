#!/usr/bin/python2 -utt
#-*- coding: utf-8 -*-

from __future__ import division, print_function
import sys
print("program begin")
from PIL import Image
from numpy.fft import fft,ifft
from copy import deepcopy
import math


import argparse
import torch
import torch.nn.init
import torch.nn as nn

import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import cv2
import copy
import PIL

from EvalMetrics import ErrorRateAt95Recall
from CDbinlosses import QuantilizeLoss, Even_distributeLoss, CorrelationPenaltyLoss
from Losses import loss_HardNet, loss_random_sampling, loss_L2Net, global_orthogonal_regularization
from W1BS import w1bs_extract_descs_and_save
from Utils import L2Norm
from Utils import str2bool
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from get_network_model import get_network_model
# Training settings
parser = argparse.ArgumentParser(description='PyTorch HardNet')
# Model options

parser.add_argument('--outnum', type=int, default=128,
                    help='output number of the descriptors')
parser.add_argument('--rotate-and-combine', type=str2bool, default=False,
                    help='my combine idea for rotate')
parser.add_argument('--savedir', type=str,
                    default='data/featmap/',
                    help='path to save feature map')
parser.add_argument('--running_mode', type=str,
                    default='training',
                    help='training, testing')
parser.add_argument('--w1bsroot', type=str,
                    default='data/sets/wxbs-descriptors-benchmark/code/',
                    help='path to dataset')
parser.add_argument('--dataroot', type=str,
                    default='data/sets/',
                    help='path to dataset')
parser.add_argument('--enable-logging',type=str2bool, default=True,
                    help='output to tensorlogger')
parser.add_argument('--log-dir', default='data/logs/',
                    help='folder to output log')
parser.add_argument('--model-dir', default='data/models/',
                    help='folder to output model checkpoints')
parser.add_argument('--experiment-name', default= 'liberty_train/',
                    help='experiment path')
parser.add_argument('--training-set', default= 'liberty',
                    help='Other options: notredame, yosemite')
parser.add_argument('--loss', default= 'triplet_margin',
                    help='Other options: softmax, contrastive')
parser.add_argument('--batch-reduce', default= 'min',
                    help='Other options: average, random, random_global, L2Net')
parser.add_argument('--num-workers', default= 8, type=int,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory',type=bool, default= True,
                    help='')
parser.add_argument('--decor',type=str2bool, default = False,
                    help='L2Net decorrelation penalty')
parser.add_argument('--anchorave', type=str2bool, default=False,
                    help='anchorave')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--mean-image', type=float, default=0.443728476019,
                    help='mean of train dataset for normalization')
parser.add_argument('--std-image', type=float, default=0.20197947209,
                    help='std of train dataset for normalization')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--end-epoch', default=10, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--anchorswap', type=str2bool, default=True,
                    help='turns on anchor swap')
parser.add_argument('--batch-size', type=int, default=1024, metavar='BS',
                    help='input batch size for training (default: 1024)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='BST',
                    help='input batch size for testing (default: 1024)')
parser.add_argument('--n-triplets', type=int, default=5000000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')
parser.add_argument('--gor',type=str2bool, default=False,
                    help='use gor')
parser.add_argument('--testbinary',type=str2bool, default=False,
                    help='test binary')
parser.add_argument('--quan_scale',type=float, default=1.0,
                    help='quan scale')
parser.add_argument('--quan', type=str2bool, default=False,
                    help='use quan')
parser.add_argument('--evendis', type=str2bool, default=False,
                    help='use evendis')
parser.add_argument('--freq', type=float, default=10.0,
                    help='frequency for cyclic learning rate')
parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                    help='gor parameter')
parser.add_argument('--quan_weights', type=float, default=1.0, metavar='ALPHA',
                    help='quan parameter')
parser.add_argument('--even_weights', type=float, default=1.0, metavar='ALPHA',
                    help='even parameter')
parser.add_argument('--cor_weights', type=float, default=1.0, metavar='ALPHA',
                    help='cor parameter')
parser.add_argument('--lr', type=float, default=10.0, metavar='LR',
                    help='learning rate (default: 10.0. Yes, ten is not typo)')
parser.add_argument('--constantlr', type=str2bool, default=False,
                    help='learning rate is a constant')
parser.add_argument('--fliprot', type=str2bool, default=True,
                    help='turns on flip and 90deg rotation augmentation')
parser.add_argument('--augmentation', type=str2bool, default=False,
                    help='turns on shift and small scale rotation augmentation')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: SGD)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')
parser.add_argument('--newstart', type=str, default= '',
                    help='newstart')
parser.add_argument('--network', default= 'L2NET',
                    help='Other options: L2NET, L2Net_channelwise, Achannelwise, Axchannelwise,G,H,I')
parser.add_argument('--pretrained', type=str2bool, default= 'False',
                    help='To define if the network is pretrained')
args = parser.parse_args()

args.epochs= args.end_epoch - args.start_epoch
# reshape image
np_reshape = lambda x: np.reshape(x, (args.imageSize, args.imageSize, 1))

cv2_scale = lambda x: cv2.resize(x, dsize=(args.imageSize, args.imageSize),
                                 interpolation=cv2.INTER_LINEAR)
suffix = '{}_{}_{}_{}_lr{}_bs{}'.format(args.network, args.training_set, args.batch_reduce,args.margin,args.lr,args.batch_size)
print("experiment name is : ",args.experiment_name)
if args.gor:
    suffix = suffix+'_gor_alpha{:1.1f}'.format(args.alpha)
if args.quan:
    suffix = suffix+'_quan_weights{:f}scale{:f}'.format(args.quan_weights,args.quan_scale)
if args.evendis:
    suffix = suffix+'_evendis_weights{:f}'.format(args.even_weights)
if args.decor:
    suffix = suffix+'_cor_weights{:f}'.format(args.cor_weights)
if args.anchorswap:
    suffix = suffix + '_as'
if args.anchorave:
    suffix = suffix + '_av'
if args.fliprot:
    suffix = suffix + '_augfliprot'
if "CDbin_NET_deep" in args.network or "BCNN" in args.network:
    suffix = suffix + "len_" + str(args.outnum)
if args.pretrained:
    suffix = suffix + "_pretrained"
triplet_flag = (args.batch_reduce == 'random_global') or args.gor
newstart = args.newstart
dataset_names = ['liberty', 'notredame', 'yosemite']

TEST_ON_W1BS = False
# check if path to w1bs dataset testing module exists
if os.path.isdir(args.w1bsroot):
    sys.path.insert(0, args.w1bsroot)
    import utils.w1bs as w1bs
    TEST_ON_W1BS = True

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
gid=[]
for v in args.gpu_id.split(','):
    gid.append(int(v))
print("gpu id is :",gid)

# torch.cuda.set_device([1,2,3,4])
print("using gpu ",args.gpu_id)
torch.cuda.set_device(gid[0])
print("using gpu ",torch.cuda.current_device())

args.cuda = not args.no_cuda and torch.cuda.is_available()


if args.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

# create loggin directory
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

import torch.utils.data as data
# set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
class TotalDatasetsLoader(data.Dataset):

    def __init__(self, datasets_path, train = True, transform = None, batch_size = None, n_triplets = 5000000, fliprot = False, *arg, **kw):
        super(TotalDatasetsLoader, self).__init__()
        datasets_path = [os.path.join(datasets_path, dataset) for dataset in os.listdir(datasets_path) if '.pt' in dataset]
        datasets = [torch.load(dataset) for dataset in datasets_path]
        print (datasets_path)
        data, labels = datasets[0][0], datasets[0][1]

        for i in range(1,len(datasets)):
            data = torch.cat([data,datasets[i][0]])
            labels = torch.cat([labels, datasets[i][1]+torch.max(labels)+1])

        del datasets

        self.data, self.labels = data, labels
        self.transform = transform
        self.train = train
        self.n_triplets = n_triplets
        self.batch_size = batch_size
        self.fliprot = fliprot
        if self.train:
                print('Generating {} triplets'.format(self.n_triplets))
                self.triplets = self.generate_triplets(self.labels, self.n_triplets, self.batch_size)

    @staticmethod
    def generate_triplets(labels, num_triplets, batch_size):
            def create_indices(_labels):
                inds = dict()
                for idx, ind in enumerate(_labels):
                    if ind not in inds:
                        inds[ind] = []
                    inds[ind].append(idx)
                return inds

            triplets = []
            indices = create_indices(labels.numpy())
            unique_labels = np.unique(labels.numpy())
            n_classes = unique_labels.shape[0]
            # add only unique indices in batch
            already_idxs = set()

            for x in tqdm(range(num_triplets)):
                if len(already_idxs) >= batch_size:
                    already_idxs = set()
                c1 = np.random.randint(0, n_classes)
                while c1 in already_idxs:
                    c1 = np.random.randint(0, n_classes)
                already_idxs.add(c1)
                c2 = np.random.randint(0, n_classes)
                while c1 == c2:
                    c2 = np.random.randint(0, n_classes)
                if len(indices[c1]) == 2:  # hack to speed up process
                    n1, n2 = 0, 1
                else:
                    n1 = np.random.randint(0, len(indices[c1]))
                    n2 = np.random.randint(0, len(indices[c1]))
                    while n1 == n2:
                        n2 = np.random.randint(0, len(indices[c1]))
                n3 = np.random.randint(0, len(indices[c2]))
                triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
            return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
            def transform_img(img):
                if self.transform is not None:
                    img = img.numpy()
                    img = self.transform(img)
                return img

            t = self.triplets[index]
            a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

            img_a = transform_img(a)
            img_p = transform_img(p)

            # transform images if required
            if self.fliprot:
                do_flip = random.random() > 0.5
                do_rot = random.random() > 0.5

                if do_rot:
                    img_a = img_a.permute(0,2,1)
                    img_p = img_p.permute(0,2,1)

                if do_flip:
                    img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,:,::-1]))
                    img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,:,::-1]))
            return img_a, img_p

    def __len__(self):
            if self.train:
                return self.triplets.size(0)

class TripletPhotoTour(dset.PhotoTour):
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, train=True, transform=None, batch_size = None,load_random_triplets = False,  *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform
        self.out_triplets = load_random_triplets
        self.train = train
        self.n_triplets = args.n_triplets
        self.batch_size = batch_size

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets)

    @staticmethod
    def generate_triplets(labels, num_triplets):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels.numpy())
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        already_idxs = set()
        print(len(labels.numpy()))
        print(len(labels.numpy())*1.0/(n_classes))
        # for x in range(n_classes):
        #     print("class ",x," have ",len(indices[x]),"images")

        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= args.batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes)
            already_idxs.add(c1)
            c2 = np.random.randint(0, n_classes)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]))
                n2 = np.random.randint(0, len(indices[c1]))
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]))
            n3 = np.random.randint(0, len(indices[c2]))
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]
        # print("a",a.size())
        img_a = transform_img(a)
        img_p = transform_img(p)
        # print("img_a", img_a.size())
        img_n = None
        if self.out_triplets:
            img_n = transform_img(n)
        if args.rotate_and_combine:
            # print('r90',np.shape(np.rot90(img_a.numpy())))
            img_a = torch.from_numpy(
                (np.rot90(img_a.numpy(),axes=(1,2))+img_a.numpy()+
                     np.rot90(np.rot90(img_a.numpy(),axes=(1,2)),axes=(1,2))+
                     np.rot90(np.rot90(np.rot90(img_a.numpy(),axes=(1,2)),
                        axes=(1,2)),axes=(1,2)))/4)
            img_p = torch.from_numpy(
                (np.rot90(img_p.numpy(),axes=(1,2))+img_p.numpy()+
                     np.rot90(np.rot90(img_p.numpy(),axes=(1,2)),axes=(1,2))+
                     np.rot90(np.rot90(np.rot90(img_p.numpy(),axes=(1,2)),
                        axes=(1,2)),axes=(1,2)))/4)
            if self.out_triplets:
                img_n = torch.from_numpy(
                    (np.rot90(img_n.numpy(),axes=(1,2))+img_n.numpy()+
                     np.rot90(np.rot90(img_n.numpy(),axes=(1,2)),axes=(1,2))+
                     np.rot90(np.rot90(np.rot90(img_n.numpy(),axes=(1,2)),
                        axes=(1,2)),axes=(1,2)))/4)
            # print(img_a.size())
            args.fliprot=False
        # transform images if required
        if args.fliprot:
            do_flip = random.random() > 0.5
            do_rot = random.random() > 0.5
            if do_rot:
                img_a = img_a.permute(0,2,1)
                img_p = img_p.permute(0,2,1)
                if self.out_triplets:
                    img_n = img_n.permute(0,2,1)
            if do_flip:
                img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,:,::-1]))
                img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,:,::-1]))
                if self.out_triplets:
                    img_n = torch.from_numpy(deepcopy(img_n.numpy()[:,:,::-1]))
        if self.out_triplets:
            return (img_a, img_p, img_n)
        else:
            return (img_a, img_p)

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)


def create_test_loader():
    test_dataset_names = copy.copy(dataset_names)
    try:
        test_dataset_names.remove(args.training_set)
        print('removing'+args.training_set)
    except:
        for i in args.training_set:
            print('removing'+i)
            test_dataset_names.remove(i)
    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}
    np_reshape64 = lambda x: np.reshape(x, (64, 64, 1))
    transform_test = transforms.Compose([
            transforms.Lambda(np_reshape64),
            transforms.ToPILImage(),
            transforms.Resize(args.imageSize),
            transforms.ToTensor()])
    transform = transforms.Compose([
            transforms.Lambda(cv2_scale),
            transforms.Lambda(np_reshape),
            transforms.ToTensor(),
            transforms.Normalize((args.mean_image,), (args.std_image,))])
    if not args.augmentation:
        transform_train = transform
        transform_test = transform
    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
             TripletPhotoTour(train=False,
                     batch_size=args.test_batch_size,
                     root=args.dataroot,
                     name=name,
                     download=True,
                     transform=transform_test),
                        batch_size=args.test_batch_size,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names]
    return test_loaders

def create_loaders(load_random_triplets = False):

    test_dataset_names = copy.copy(dataset_names)
    # test_dataset_names.remove(args.training_set)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    np_reshape64 = lambda x: np.reshape(x, (64, 64, 1))
    transform_test = transforms.Compose([
            transforms.Lambda(np_reshape64),
            transforms.ToPILImage(),
            transforms.Resize(args.imageSize),
            transforms.ToTensor()])
    transform_train = transforms.Compose([
            transforms.Lambda(np_reshape64),
            transforms.ToPILImage(),
            transforms.RandomRotation(5,PIL.Image.BILINEAR),
            transforms.RandomResizedCrop(args.imageSize, scale = (0.9,1.0),ratio = (0.9,1.1)),
            transforms.Resize(args.imageSize),
            transforms.ToTensor()])
    # print("args.imageSize",args.imageSize)
    transform = transforms.Compose([
            transforms.Lambda(cv2_scale),
            transforms.Lambda(np_reshape),
            transforms.ToTensor(),
            transforms.Normalize((args.mean_image,), (args.std_image,))])
    if not args.augmentation:
        transform_train = transform
        transform_test = transform
        # print("no augmentation")
        if args.training_set=='all':
            train_loader = torch.utils.data.DataLoader(
                TotalDatasetsLoader(train=True,
                                    load_random_triplets=load_random_triplets,
                                    batch_size=args.batch_size,
                                    datasets_path=args.dataroot,
                                    fliprot=args.fliprot,
                                    n_triplets=args.n_triplets,
                                    name=args.training_set,
                                    download=True,
                                    transform=transform_train),
                batch_size=args.batch_size,
                shuffle=False, **kwargs)
        else:
            train_loader = torch.utils.data.DataLoader(
                    TripletPhotoTour(train=True,
                             load_random_triplets = load_random_triplets,
                             batch_size=args.batch_size,
                             root=args.dataroot,
                             name=args.training_set,
                             download=True,
                             transform=transform_train),
                             batch_size=args.batch_size,
                             shuffle=False, **kwargs)

    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
             TripletPhotoTour(train=False,
                     batch_size=args.test_batch_size,
                     root=args.dataroot,
                     name=name,
                     download=True,
                     transform=transform_test),
                        batch_size=args.test_batch_size,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return train_loader, test_loaders

def train(test_loader, train_loader, model, optimizer, epoch, logger, load_triplets  = False):
    # switch to train mode

    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data in pbar:
        if load_triplets:
            data_a, data_p, data_n = data
        else:
            data_a, data_p = data
        # print("data_a",data_a.size())
        if args.cuda:
            data_a, data_p  = data_a.cuda(), data_p.cuda()
            data_a, data_p = Variable(data_a), Variable(data_p)
            out_a, out_p = model(data_a), model(data_p)

        if load_triplets:
            data_n  = data_n.cuda()
            data_n = Variable(data_n)
            out_n = model(data_n)

        if args.batch_reduce == 'L2Net':
            loss = loss_L2Net(out_a, out_p, anchor_swap = args.anchorswap,
                    margin = args.margin, loss_type = args.loss)
        elif args.batch_reduce == 'random_global':
            loss = loss_random_sampling(out_a, out_p, out_n,
                margin=args.margin,
                anchor_swap=args.anchorswap,
                loss_type = args.loss)
        else:
            loss = loss_HardNet(out_a, out_p,
                            margin=args.margin,
                            anchor_swap=args.anchorswap,
                            anchor_ave=args.anchorave,
                            batch_reduce = args.batch_reduce,
                            loss_type = args.loss)

        if args.decor:
            loss += args.cor_weights * CorrelationPenaltyLoss()(out_a)
        if args.gor:
            loss += args.alpha * global_orthogonal_regularization(out_a, out_n)
        if args.evendis:
            loss += args.even_weights * Even_distributeLoss()(out_a)
        if args.quan:
            loss += args.quan_weights * QuantilizeLoss(args.quan_scale)(out_a)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not args.constantlr:
            adjust_learning_rate(optimizer)
        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]lr:{:f} \tLoss_T: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           optimizer.param_groups[0]['lr'], loss.data[0]))

    if (args.enable_logging):
        logger.log_value('loss', loss.data.item()).step()

    try:
        os.stat('{}{}'.format(args.model_dir,suffix))
    except:
        os.makedirs('{}{}'.format(args.model_dir,suffix))

    torch.save({'epoch': epoch + 1, 'optimizer':optimizer.state_dict()
                ,'state_dict': model.state_dict()},
               '{}{}/checkpoint_{}{}.pth'.format(args.model_dir,suffix,newstart,epoch))
    # torch.save(model,'{}{}/checkpoint_{}.pth'.format(args.model_dir,suffix,epoch))
    print("model {}{}/checkpoint_{}{}.pth is saved".format(args.model_dir,suffix,newstart,epoch))
    if (args.enable_logging):
        logger.log_value(test_loader['name']+'loss is:', loss.data[0])
    return loss.data.item()



def test(test_loader, model, epoch, logger, logger_test_name):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    with torch.no_grad():
        for batch_idx, (data_a, data_p, label) in pbar:

            if args.cuda:
                data_a, data_p = data_a.cuda(), data_p.cuda()

            data_a, data_p, label = Variable(data_a), \
                                    Variable(data_p), Variable(label)

            out_a, out_p = model(data_a), model(data_p)
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy().reshape(-1,1))
            ll = label.data.cpu().numpy().reshape(-1, 1)
            labels.append(ll)
            if batch_idx % args.log_interval == 0:
                pbar.set_description(logger_test_name+' Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx * len(data_a), len(test_loader.dataset),
                           100. * batch_idx / len(test_loader)))

    num_tests = test_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
    print('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))

    if (args.enable_logging):
        logger.log_value(logger_test_name+' fpr95', fpr95)
    return fpr95

def test_binary(test_loader, model, epoch, logger, logger_test_name):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    with torch.no_grad():
        for batch_idx, (data_a, data_p, label) in pbar:

            if args.cuda:
                data_a, data_p = data_a.cuda(), data_p.cuda()

            data_a, data_p, label = Variable(data_a), \
                                    Variable(data_p), Variable(label)

            out_a, out_p = torch.sign(model(data_a)), torch.sign(model(data_p))
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy().reshape(-1,1))
            ll = label.data.cpu().numpy().reshape(-1, 1)
            labels.append(ll)

            if batch_idx % args.log_interval == 0:
                pbar.set_description(logger_test_name+' Test(binary) Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx * len(data_a), len(test_loader.dataset),
                           100. * batch_idx / len(test_loader)))

        num_tests = test_loader.dataset.matches.size(0)
        labels = np.vstack(labels).reshape(num_tests)
        distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
    print('\33[91mTest(binary) set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))

    if (args.enable_logging):
        logger.log_value(logger_test_name+'(binary) fpr95', fpr95)
    return fpr95

def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = args.lr * (
                1.0 - float(group['step']) * 
	        float(args.batch_size) / (args.n_triplets * float(args.epochs)))
    return

def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=new_lr,
                               weight_decay=args.wd)
    else:
        raise Exception('Not supported optimizer: {0}'.format(args.optimizer))
    return optimizer

def test_model(test_loaders, model, logger):
    print('\nparsed options:\n{}\n'.format(vars(args)))
    if args.cuda:
        model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                t=0
                for i in model.state_dict():
                    if(i.startswith('features') or i.startswith('feature_L2NET')):
                        model.state_dict()[i]=checkpoint['state_dict'].items()[t]
                        t=t+1
                        print(i+" is loaded")
                args.start_epoch = 0
            # model = torch.load(args.resume)
        else:
            print('=> no checkpoint found at {}'.format(args.resume))
    epoch=args.start_epoch
    for test_loader in test_loaders:
        test(test_loader['dataloader'], model, epoch, logger, test_loader['name'])
        if args.testbinary:
            test_binary(test_loader['dataloader'], model, epoch, logger, test_loader['name'])
     


def main(train_loader, test_loaders, model, logger, file_logger):
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))

    #if (args.enable_logging):
    #    file_logger.log_string('logs.txt', '\nparsed options:\n{}\n'.format(vars(args)))

    if args.cuda:
        model.cuda()
    # change model.features to model.parameters()
    optimizer1 = create_optimizer(model, args.lr)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                t=0
                for i in model.state_dict():
                    if(i.startswith('features') or i.startswith('feature_L2NET')):
                        model.state_dict()[i]=checkpoint['state_dict'].items()[t]
                        t=t+1
                        print(i+" is loaded")
                args.start_epoch=0

            try:
                optimizer1.load_state_dict(checkpoint['optimizer'])
                print('=> optimizer loaded from {}'.format(args.resume))
                print('current lr is {}'.format(optimizer1.param_groups[0]['lr']))
                for group in optimizer1.param_groups:
                   if 'step' not in group:
                       group['step'] = 0.
                   else:
                       print('current step is {}'.format(group['step']))
                #        group['step'] = (args.n_triplets)*args.start_epoch//args.batch_size+1
            except:
                print('=> on optimizer saved in {}'.format(args.resume))
        else:
            print('=> no checkpoint found at {}'.format(args.resume))
            
    
    start = args.start_epoch
    end = args.end_epoch if args.end_epoch < (start + args.epochs) else (start + args.epochs)
    trainloss=[]
    testacc=[]
    for epoch in range(start, end):

        # iterate over test loaders and test results
        train(test_loaders[0], train_loader, model, optimizer1, epoch, logger, triplet_flag)
        for test_loader in test_loaders:
            test(test_loader['dataloader'], model, epoch, logger, test_loader['name'])
            if args.testbinary:
                test_binary(test_loader['dataloader'], model, epoch, logger, test_loader['name'])
        x = copy.copy(dataset_names)
        x.remove(args.training_set)
        old_trainingset=args.training_set
        args.training_set = x[0]
        args.training_set = old_trainingset
        if TEST_ON_W1BS :
            # print(weights_path)
            patch_images = w1bs.get_list_of_patch_images(
                DATASET_DIR=args.w1bsroot.replace('/code', '/data/W1BS'))
            desc_name = 'curr_desc'# + str(random.randint(0,100))
            
            DESCS_DIR = LOG_DIR + '/temp_descs/' #args.w1bsroot.replace('/code', "/data/out_descriptors")
            OUT_DIR = DESCS_DIR.replace('/temp_descs/', "/out_graphs/")

            for img_fname in patch_images:
                w1bs_extract_descs_and_save(img_fname, model, desc_name, cuda = args.cuda,
                                            mean_img=args.mean_image,
                                            std_img=args.std_image, out_dir = DESCS_DIR)


            force_rewrite_list = [desc_name]
            w1bs.match_descriptors_and_save_results(DESC_DIR=DESCS_DIR, do_rewrite=True,
                                                    dist_dict={},
                                                    force_rewrite_list=force_rewrite_list)
        train_loader, test_loaders2 = create_loaders(load_random_triplets=triplet_flag)


    


if __name__ == '__main__':
    print("program begin")
    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    LOG_DIR = os.path.join(args.log_dir, suffix)
    DESCS_DIR = os.path.join(LOG_DIR, 'temp_descs')
    if TEST_ON_W1BS:
        if not os.path.isdir(DESCS_DIR):
            os.makedirs(DESCS_DIR)
    logger, file_logger = None, None
    model = get_network_model(args.network, args.outnum, pretrained=args.pretrained)

    model = nn.DataParallel(model, device_ids=gid)
    if(args.enable_logging):
        from Loggers import Logger, FileLogger
        logger = Logger(LOG_DIR)
        #file_logger = FileLogger(./log/+suffix)



    if(args.running_mode=='testing'):
        print(("NOT " if not args.cuda else "") + "Using cuda")
        test_loaders = create_test_loader()
        test_model(test_loaders, model, logger)
    elif(args.running_mode=='get_checkpoint_name'):
        print("get_checkpoint_name:{}{}/checkpoint_{}".format(args.model_dir, suffix, newstart))
    else:
        print(("NOT " if not args.cuda else "") + "Using cuda")
        train_loader, test_loaders = create_loaders(load_random_triplets = triplet_flag)
        main(train_loader, test_loaders, model, logger, file_logger)
