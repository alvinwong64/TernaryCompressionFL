'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.multiprocessing as mp

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import json
import random
import numpy as np

from models import *
from utils import progress_bar


def random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def acc_topk(preds, targets, k=5):
    with torch.no_grad():
        total = 0
        top1 = 0
        topk = 0
        _, maxk = torch.topk(preds, k, dim=-1)
        total += targets.size(0)
        targets = targets.view(-1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]
        top1 += (targets == maxk[:, 0: 1]).sum()
        topk += (targets == maxk).sum()
    return top1, topk


# Training
def train(args, net, device, trainloader, train_loss_, train_acc1_, train_acc5_, optimizer, criterion, epoch, rank,
          his):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    # correct = 0
    acc1 = 0
    acc5 = 0
    total = 0
    batch_loss = []

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(rank), targets.to(rank)
        inputs = inputs.half()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        dist.all_reduce(loss, op=dist.ReduceOp.SUM)

        train_loss += loss.item()
        # _, predicted = outputs.max(1)
        total += targets.size(0)
        total_t = torch.tensor(total).to(rank)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
        total_t = total_t.item()
        top1, top5 = acc_topk(outputs, targets, args.topk)
        dist.all_reduce(top1, op=dist.ReduceOp.SUM)
        dist.all_reduce(top5, op=dist.ReduceOp.SUM)
        acc1 += top1.item()
        acc5 += top5.item()
        # correct += predicted.eq(targets).sum().item()
        batch_loss.append(loss.item())
        if (rank == 0):
            progress_bar(batch_idx, len(trainloader), '%s Loss: %.3f | top1: %.3f%% top5: %.3f%% (%d/%d)'
                         % (his, train_loss / (batch_idx + 1), 100. * acc1 / total_t ,
                            100. * acc5 / total_t , acc1, total_t ))
    train_loss_.append(train_loss / (len(trainloader.dataset) / 128))
    train_acc1_.append(acc1 / len(trainloader.dataset))
    train_acc5_.append(acc5 / len(trainloader.dataset))


def test(args, net, device, testloader, test_loss_, test_acc1_, test_acc5_, criterion, best_acc, epoch, rank):
    net.eval()
    test_loss = 0
    # correct = 0
    acc1 = 0
    acc5 = 0
    total = 0
    batch_loss = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(rank), targets.to(rank)
            inputs = inputs.half()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            test_loss += loss.item()
            total += targets.size(0)
            total_t = torch.tensor(total).to(rank)
            dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
            total_t = total_t.item()
            top1, top5 = acc_topk(outputs, targets, args.topk)
            dist.all_reduce(top1, op=dist.ReduceOp.SUM)
            dist.all_reduce(top5, op=dist.ReduceOp.SUM)
            acc1 += top1.item()
            acc5 += top5.item()
            # correct += predicted.eq(targets).sum().item()
            batch_loss.append(loss.item())
            if (rank == 0):
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | top1: %.3f%% top5: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * acc1 / total_t ,
                                100. * acc5 / total_t , acc1, total_t ))
            # _, predicted = outputs.max(1)
            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()

        test_loss_.append(test_loss / (len(testloader.dataset) / 100))
        test_acc1_.append(acc1 / total_t)
        test_acc5_.append(acc5 / total_t )

    # Save checkpoint.
    acc = 100. * acc1 / total * args.GPUS
    if (rank == 0):
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/{}.ckpt'.format(args.his))
            return acc
        else:
            return best_acc


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method='env://')


def cleanup():
    dist.destroy_process_group()


def distributed_training(rank, world_size, args, use_cuda):
    # --------------- Setup -------------#
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    # -----------------------------------#

    random_seed(80)

    device = torch.device("cuda" if use_cuda else "cpu")

    # ---------------- Data ------------#
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print('==> Preparing ImageNet data..')
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # ImageNet
    trainset = torchvision.datasets.ImageFolder(
        root='/data/datasets/ILSVRC2012/train', transform=transform_train)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        num_replicas=world_size,
        rank=rank
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler)

    testset = torchvision.datasets.ImageFolder(
        root='/data/datasets/ILSVRC2012/val', transform=transform_test)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        testset,
        num_replicas=world_size,
        rank=rank
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True, sampler=test_sampler)

    # -----------------------------------#
    # Model
    print('==> Building model..')

    Model = {
        'google_tnt': GoogLeNet_tnt,
        'google_norm': GoogLeNet
    }

    net = Model[args.model](aux_logits=False).half().to(rank)
    net = DDP(net, device_ids=[rank])

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

        checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=torch.device(rank))
        net.load_state_dict(checkpoint['net'], strict=False)
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_loss_ = []
    train_acc1_ = []
    train_acc5_ = []
    test_loss_ = []
    test_acc1_ = []
    test_acc5_ = []

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(args, net, device, trainloader, train_loss_, train_acc1_, train_acc5_, optimizer, criterion, epoch, rank,
              args.his)
        best_acc = test(args, net, device, testloader, test_loss_, test_acc1_, test_acc5_, criterion, best_acc, epoch,
                        rank)
        scheduler.step()

    his_dict = {
        'train_loss': train_loss_,
        'train_top1': train_acc1_,
        'train_top5': train_acc5_,
        'test_loss': test_loss_,
        'test_top1': test_acc1_,
        'test_top5': test_acc5_,
        #     'train_time': train_time,
        #     'glob_zero_rates': comp_rate,
        #     'local_zero_rates': update_zero_rate,

    }
    if (rank == 0):
        os.makedirs('./his/', exist_ok=True)
        with open('./his/{}.json'.format(args.his), 'w+') as f:
            json.dump(his_dict, f, indent=2)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--model', default='res18_norm', type=str)
    parser.add_argument('--GPUS', default=4, type=int, help='number of gpus')
    parser.add_argument('--his', type=str, required=True)
    parser.add_argument('--epochs', default=100, type=int, help='the total epochs of training')
    parser.add_argument('--topk', default=5, type=int, help='the top1 and topk acc')
    parser.add_argument('--class_num', default=1000, type=int, help='number of class')
    parser.add_argument('--batch', default=128, type=int, help='batch size for per iteration')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--local_rank", type=int, default=4)
    args = parser.parse_args()

    args_dict = vars(args)

    if not os.path.isdir('setting'):
        os.mkdir('setting')

    with open('./setting/config_{}.json'.format(args.his), 'w+') as f:
        json.dump(args_dict, f)

    world_size = args.GPUS
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    mapclass = {}
    with open('./class_name/map_clsloc.txt') as f:
        for line in f:
            dname, idx, classname = line.split(' ')
            mapclass[dname] = classname
    classes = [mapclass[dname] for dname in os.listdir('/data/datasets/ILSVRC2012') if dname in mapclass]
    distributed_training(args.local_rank,world_size, args, use_cuda)
    # if torch.cuda.device_count() > 1:
    #     print("We have available ", torch.cuda.device_count(), "GPUs! but using ", world_size, " GPUs")
    #
    # mp.spawn(distributed_training, args=(world_size, args, use_cuda), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
