'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.multiprocessing as mp
from tools_distributed import *
import torchvision
import torchvision.transforms as transforms
import time
import os
import argparse
import json
import random
import numpy as np

from models import *


def random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    if args.dataset == 'imagenet':
        print('==> Preparing ImageNet data..')
        transform_train = transforms.Compose([
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
        dataset_train = torchvision.datasets.ImageFolder(
            root='/data/datasets/ILSVRC2012/train', transform=transform_train)

        dataset_test = torchvision.datasets.ImageFolder(
            root='/data/datasets/ILSVRC2012/val', transform=transform_test)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test,
            num_replicas=world_size,
            rank=rank
        )

    else:
        print('==> Preparing CIFAR10 data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            #         transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            #         transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # CIFAR10

        dataset_train = torchvision.datasets.CIFAR10(
            root='/data/datasets/cifar10', train=True, download=True, transform=transform_train)

        dataset_test = torchvision.datasets.CIFAR10(
            root='/data/datasets/cifar10', train=False, download=True, transform=transform_test)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test,
            num_replicas=world_size,
            rank=rank
        )

    if args.noniid:
        # train_dataset, test_dataset, num_users, n_class, num_samples, rate_unbalance
        dict_users_train, dict_users_test = cifar_extr_noniid(dataset_train, dataset_test,
                                                              args.num_users, args.n_class)
    else:
        dict_users_train = iid(dataset_train, args.num_users)

    # -----------------------------------#
    # Model
    print('==> Building model..')
    Model = {
        'vgg_tnt': VGG_tnt,
        'vgg_norm': VGG_norm,
        'mobilev2_tnt': MobileNetV2_tnt,
        'mobilev2_norm': MobileNetV2,
        'res18_tnt': ResNet_TNT18,
        'res50_tnt': ResNet_TNT50,
        'res18_norm': ResNet18,
        'res50_norm': ResNet50,
        'alex_tnt': AlexNet_tnt,
        'alex_norm': AlexNet,
        'google_tnt': GoogLeNet_tnt,
        'google_norm': GoogLeNet
    }

    if args.model == 'google_tnt':
        net_glob = Model[args.model](aux_logits=False).half().to(rank)
    elif args.model == 'google_norm':
        net_glob = Model[args.model](aux_logits=False).half().to(rank)
    else:
        net_glob = Model[args.model](args.class_num).half().to(rank)
    if rank ==0:
        print(net_glob)
    net_glob = DDP(net_glob, device_ids=[rank])
    net_glob.modules()

    # training
    current_lr = args.lr
    best_acc = 0
    glob_agg_num = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    train_acc, train_loss = [], []
    test_acc, test_loss = [], []
    train_time = []
    comp_rate = []
    update_zero_rate = []

    client_net = Client_net(net_glob, args.num_users)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        start_time = time.time()
        client_upload = {}
        client_local = {}
        acc_locals_train = {}
        loss_locals_train = []
        local_zero_rates = []
        if (rank == 0):
            print(f'c\n | Global Training Round: {epoch} Training {args.his}|\n')
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # training
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx],
                                client=idx, rank=rank, world_size=world_size) if not args.noniid else LocalUpdate(args=args,
                                                                                                           dataset=dataset_train,
                                                                                                           idxs=np.int_(
                                                                                                               dict_users_train[
                                                                                                                   idx]),
                                                                                                           client=idx,
                                                                                                           rank=rank,
                                                                                                           world_size=world_size)
            network, loss_local_train, acc_local_train = local.train(net=client_net[str(idx)], rank=rank, lr=current_lr)
            # Global TNT weights or Norm Weights
            if args.tntupload:
                # if (epoch+1) % args.g_c == 0:
                #     print('floating update')
                #     client_upload[str(idx)] = copy.deepcopy(client_net[str(idx)].state_dict())
                #     local_zero_rates.append(0)
                # else:
                if rank == 0:
                    print('ternary update')
                w_tnt, local_error = ternary_convert(copy.deepcopy(client_net[str(idx)]))
                client_local[str(idx)] = copy.deepcopy(local_error)
                client_upload[str(idx)] = copy.deepcopy(w_tnt)
                z_r = zero_rates(w_tnt)
                local_zero_rates.append(z_r)
                if rank == 0:
                    print('Client {} zero rate {:.2%}'.format(idx, z_r))
            else:
                client_upload[str(idx)] = copy.deepcopy(client_net[str(idx)].state_dict())

            # recording local training info
            acc_locals_train[str(idx)] = copy.deepcopy(acc_local_train)
            loss_locals_train.append(copy.deepcopy(loss_local_train))
        elapsed = time.time() - start_time
        train_time.append(elapsed)

        glob_avg, cr = FedAvg(copy.deepcopy(client_upload), copy.deepcopy(acc_locals_train), 1)
        print('Global Zero Rates {:.2%}'.format(cr))
        comp_rate.append(cr)

        # update local models
        if args.tntupload:
            # if (epoch+1) % args.g_c == 0:
            #     glob_agg_num += 1
            #     print('floating update')
            #     for idx in idxs_users:
            #         client_net[str(idx)].load_state_dict(glob_avg)
            # else:
            for idx in idxs_users:
                client_net[str(idx)] = rec_w(copy.deepcopy(glob_avg),
                                             copy.deepcopy(client_local[str(idx)]),
                                             client_net[str(idx)])
        else:
            for idx in idxs_users:
                client_net[str(idx)].load_state_dict(glob_avg)

        # Testing

        print(f'\n |Round {epoch} Global Test {args.his}|\n')
        client_acc = []
        client_loss = []
        for idx in idxs_users:
            acc_t, loss_t, best_acc = test_img(idx, epoch, client_net[str(idx)],
                                               dataset_test, args, best_acc, sampler=test_sampler, rank=rank)
            client_acc.append(acc_t)
            client_loss.append(loss_t)
        test_acc.append(sum(client_acc) / len(idxs_users))
        test_loss.append(sum(client_loss) / len(idxs_users))

        # training info update
        avg_acc_train = sum(acc_locals_train.values()) / len(acc_locals_train.values())
        #     print(train_acc)
        train_acc.append(avg_acc_train)
        #     try:
        #         print('[INFO] acc. rate', abs((train_acc[-1] - train_acc[-2]) / (train_acc[-2] - train_acc[-3])))
        #     except:
        #         pass

        loss_avg = sum(loss_locals_train) / len(loss_locals_train)
        train_loss.append(loss_avg)
        try:
            temp_zero_rates = sum(local_zero_rates) / len(local_zero_rates)
        except:
            temp_zero_rates = sum(local_zero_rates)
        update_zero_rate.append(temp_zero_rates)

        #     writer.add_scalar("Loss/train", loss, epoch)
        #     writer.flush()
        if rank == 0:
            print('Round {} costs time: {:.2f}s| Train Acc.: {:.2%}| '
                  'Test Acc.{:.2%}| Train loss: {:.4f}| Test loss: {:.4f}| '
                  'Down Rate is {:.3%}| Up Rate{:.3%}'
                  ' Floating agg {}'.format(
                epoch,
                elapsed,
                avg_acc_train,
                test_acc[-1],
                loss_avg,
                test_loss[-1],
                cr,
                temp_zero_rates,
                glob_agg_num
            ))

            current_lr = current_learning_rate(epoch, current_lr, args)
    if rank == 0:
        his_dict = {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_correct': test_acc,
            'train_time': train_time,
            'glob_zero_rates': comp_rate,
            'local_zero_rates': update_zero_rate,
        }

        os.makedirs('./his/', exist_ok=True)
        with open('./his/{}.json'.format(args.his), 'w+') as f:
            json.dump(his_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/Imagenet Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='the total epochs of training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--frac', default=1, type=int)
    parser.add_argument('--local_bs', default=128, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--split', default='user')
    parser.add_argument('--local_ep', default=1, type=int)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--d_epoch', default=50, type=int)
    parser.add_argument('--decay_r', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument("--local_rank", type=int, default=4, help='Will get the value from launch script')

    # Experiment of ternary image, could ignore this
    parser.add_argument('--tnt_image', action='store_true', help='ternary image experiment')

    # Usually have to parse in these arguments
    parser.add_argument('--his', type=str, required=True)
    parser.add_argument('--model', default='res18_norm', type=str)
    parser.add_argument('--GPUS', default=4, type=int, help='number of gpus')
    parser.add_argument('--pickgpu', default="0,1,2,3", type=str, help='pick which gpus on the machine to use')
    parser.add_argument('--save', action='store_true', help='save model every 10 epoch')
    parser.add_argument('--num_users', default=10, type=int)
    parser.add_argument('--dataset', default='imagenet', help='pick dataset imagenet/cifar10')
    parser.add_argument('--class_num', default=1000, type=int, help='total number of class')
    parser.add_argument('--tntupload', action='store_true', help='uploading tnt weights')
    parser.add_argument('--noniid', action='store_true', help='call this argument if want to use noniid dataset')
    parser.add_argument('--n_class', default=2, type=int, help='class number in each client, use for noniid cases')

    # Have to use with torch.distributed.launch script
    # Sample terminal command:
    ''' 1 machine , 4 gpus
        $   python -m torch.distributed.launch --nnode=1 --node_rank=0 --nproc_per_node=4
            tnt_fl_distributed_training.py --his googlenet_experiment --model google_tnt  --GPUS 4 --pickgpu 0,1,2,3 
            --save --dataset cifar10 --class_num 10 --tntupload --noniid --n_class 5 

        nnode = How many machine you have
        node_rank = Usually for multiple machine, eg second machine node_rank = 1
        nproc_per_node = Refer to how many process in each node, usually same with number of gpu

        For more detail :   https://github.com/pytorch/examples/tree/master/distributed/ddp
                            https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py
    '''

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.pickgpu
    args_dict = vars(args)

    if not os.path.isdir('setting'):
        os.mkdir('setting')

    with open('./setting/config_{}.json'.format(args.his), 'w+') as f:
        json.dump(args_dict, f)

    world_size = args.GPUS
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    distributed_training(args.local_rank, world_size, args, use_cuda)


if __name__ == '__main__':
    main()
