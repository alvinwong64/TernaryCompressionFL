import numpy as np
import torch.optim as optim
import torch
from models import TNT
from models.convert_tnt import *
from utils import *
from torch.utils.data import DataLoader, Dataset
import copy
import time
import os
import sys
import math
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist


def iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)  # numbers of items in one client
    dict_users, all_idxs = {}, [i for i in range(
        len(dataset))]  # dict_user for recording the client number; all_idxs is the indx of each item
    for i in range(num_users):  # choosing training data for each client
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))  # randomly choose ##num_items## items for a client without replacing
        all_idxs = list(set(all_idxs) - dict_users[i])  # remove seleceted items
    return dict_users  # contains the training data for each client i


def cifar_extr_noniid(train_dataset, test_dataset, num_users, n_class):
    num_shards_train = num_users * n_class  # minimum shard needed
    num_classes = 10
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000
    assert (n_class * num_users <= num_shards_train)
    assert (n_class <= num_classes)
    idx_class = [i for i in range(num_classes)]

    # Generate the minimum whole dataset that is needed for training
    # eg: class 5 client 5, 25 shards, minimum whole dataset = 3
    idx_shard = [i % 10 for i in range(math.ceil(
        num_shards_train / 10) * 10)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(50000)  # 5000 sample per class
    # labels = dataset.train_labels.numpy()
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)
    # labels_test_raw = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    # print(idxs_labels_test[0, :])
    # print(idxs_labels_test[1, :])

    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        # randomly pick non-repetitive classes from the dataset
        rand_set = set(np.random.choice(np.unique(idx_shard), n_class, replace=False))
        # print(idx_shard)
        # print(rand_set)
        for rand in rand_set:
            # Remove the first occurance of the chosen class
            idx_shard.remove(rand)

            # Get all samples from each class that has been chosen
            dict_users_train[i] = np.concatenate(
                (dict_users_train[i], idxs[rand * 5000:(rand + 1) * 5000]), axis=0)
            user_labels = np.concatenate((user_labels, labels[rand * 5000:(rand + 1) * 5000]),
                                         axis=0)
        # print((dict_users_train[i][0]))
        user_labels_set = set(user_labels)
        # print(user_labels_set)
        # print(user_labels)
        for label in user_labels_set:
            # print(label)
            dict_users_test[i] = np.concatenate(
                (dict_users_test[i], idxs_test[int(label) * num_imgs_perc_test:int(label + 1) * num_imgs_perc_test]),
                axis=0)
        # print(set(labels_test_raw[dict_users_test[i].astype(int)]))

    return dict_users_train, dict_users_test

    # Previous non iid method

    # def cifar_extr_noniid(train_dataset, test_dataset, num_users, n_class, num_samples, rate_unbalance):
    #
    #     num_shards_train, num_imgs_train = int(50000 / num_samples), num_samples
    #     num_classes = 10
    #     num_imgs_perc_test, num_imgs_test_total = 1000, 10000
    #     assert (n_class * num_users <= num_shards_train)
    #     assert (n_class <= num_classes)
    #     idx_class = [i for i in range(num_classes)]
    #     idx_shard = [i for i in range(num_shards_train)]
    #     dict_users_train = {i: np.array([]) for i in range(num_users)}
    #     dict_users_test = {i: np.array([]) for i in range(num_users)}
    #     idxs = np.arange(num_shards_train * num_imgs_train)
    #     # labels = dataset.train_labels.numpy()
    #     labels = np.array(train_dataset.targets)
    #     idxs_test = np.arange(num_imgs_test_total)
    #     labels_test = np.array(test_dataset.targets)
    #     # labels_test_raw = np.array(test_dataset.targets)
    #
    #     # sort labels
    #     idxs_labels = np.vstack((idxs, labels))
    #     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    #     idxs = idxs_labels[0, :]
    #     labels = idxs_labels[1, :]
    #
    #     idxs_labels_test = np.vstack((idxs_test, labels_test))
    #     idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    #     idxs_test = idxs_labels_test[0, :]
    #     # print(idxs_labels_test[0, :])
    #     # print(idxs_labels_test[1, :])
    #
    #     # divide and assign
    #     for i in range(num_users):
    #         user_labels = np.array([])
    #         rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
    #         idx_shard = list(set(idx_shard) - rand_set)
    #         unbalance_flag = 0
    #         for rand in rand_set:
    #             if unbalance_flag == 0:
    #                 dict_users_train[i] = np.concatenate(
    #                     (dict_users_train[i], idxs[rand * num_imgs_train:(rand + 1) * num_imgs_train]), axis=0)
    #                 user_labels = np.concatenate((user_labels, labels[rand * num_imgs_train:(rand + 1) * num_imgs_train]),
    #                                              axis=0)
    #             else:
    #                 dict_users_train[i] = np.concatenate(
    #                     (dict_users_train[i], idxs[rand * num_imgs_train:int((rand + rate_unbalance) * num_imgs_train)]),
    #                     axis=0)
    #                 user_labels = np.concatenate(
    #                     (user_labels, labels[rand * num_imgs_train:int((rand + rate_unbalance) * num_imgs_train)]), axis=0)
    #             # print(i, user_labels)
    #             unbalance_flag = 1
    #         user_labels_set = set(user_labels)
    #         # print(user_labels_set)
    #         # print(user_labels)
    #         for label in user_labels_set:
    #             dict_users_test[i] = np.concatenate(
    #                 (dict_users_test[i], idxs_test[int(label) * num_imgs_perc_test:int(label + 1) * num_imgs_perc_test]),
    #                 axis=0)
    #         # print(set(labels_test_raw[dict_users_test[i].astype(int)]))
    #     #     print(i,len(dict_users_test[i]))
    #     # print(len(dict_users_test))
    #
    #     return dict_users_train, dict_users_test


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, np.int_(label)


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, client=None, rank=None, world_size=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.sampler = torch.utils.data.distributed.DistributedSampler(
            DatasetSplit(dataset, idxs),
            num_replicas=world_size,
            rank=rank
        )
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=False,
                                    num_workers=0, pin_memory=True, sampler=self.sampler)
        self.client = client

    def train(self, net, rank,lr):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        #         optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)

        epoch_loss = []
        epoch_acc = []
        total = 0
        if rank== 0:
            print('Client {} is training on GPU {}.'.format(self.client, self.args.pickgpu))
        for i in range(self.args.local_ep):
            batch_loss = []
            batch_acc = 0
            correct = 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # print(images[0])
                if self.args.tnt_image:
                    images = TNT.image_tnt(images)
                images = images.to(rank).half()
                labels = labels.to(rank)
                net.zero_grad()
                log_probs = net(images)
                # print(type(log_probs))
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                # _, predicted = outputs.max(1)
                total += labels.size(0)
                total_t = torch.tensor(total).to(rank)
                dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
                total_t = total_t.item()

                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct_t = y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum().to(rank)
                dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
                correct += correct_t.item()
                train_acc = correct / len(self.ldr_train.dataset)
                batch_acc = train_acc
                batch_loss.append(loss.item())
                if (rank == 0):
                    progress_bar(batch_idx, len(self.ldr_train), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                                 % (sum(batch_loss) / (batch_idx + 1), train_acc * 100., correct,
                                    len(self.ldr_train.dataset)))

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(batch_acc)

        return net, sum(epoch_loss) / len(epoch_loss), epoch_acc[-1]


def test_img(idxs, epoch, net_g, datatest, args, best_acc, sampler=None, rank=None):
    net_g.eval()

    # testing
    test_loss = 0
    correct = 0
    total_t = 0

    data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=True,
                             sampler=sampler)
    if rank == 0:
        print('Client {} Testing on GPU {}.'.format(idxs, args.pickgpu))
    for idx, (data, target) in enumerate(data_loader):
        if args.tnt_image:
            data = TNT.image_tnt(data)
        data = data.to(rank).half()
        target = target.to(rank)
        log_probs = net_g(data)

        # sum up batch loss
        loss = F.cross_entropy(log_probs, target,reduction='sum')
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        test_loss += loss.item()

        total_t += target.size(0)
        total = torch.tensor(total_t).to(rank)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        total = total.item()

        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct_t = y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum().to(rank)
        dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
        correct += correct_t.item()
        if rank==0:
            progress_bar(idx, len(data_loader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                         % (test_loss / total, 100. * correct / total, correct, total))

    test_loss /= len(data_loader.dataset)
    acc = correct/ len(data_loader.dataset)
    w_tnt,w_error= ternary_convert(copy.deepcopy(net_g))

    # saving best
    if rank==0:
        print(f"Test loss: {test_loss} \n Test accuracy: {acc} \n")
        if acc > best_acc:
            print('Saving..')
            state = {
                # 'net': net_g.get_tnt(),  # net_g.get_tnt(),  # 'net':net.get_tnt() for tnt network // net.state_dict()
                'net': w_tnt if args.tntupload else net_g.state_dict(),
                # net_g.get_tnt(),  # 'net':net.get_tnt() for tnt network // net.state_dict()
                'acc': acc * 100.,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/{}.ckpt'.format(args.his))
            best_acc = acc

        if args.save:
            dict_name = args.his.split('.')[0]
            path = os.path.join('./saved/', '{}/epoch_{}_{}.ckpt'.format(dict_name, epoch, args.his))
            if epoch % 10 == 0:
                print('Saving..')
                state = {
                    # 'net': net_g.get_tnt(),
                    'net': w_tnt if args.tntupload else net_g.state_dict(),
                    # net_g.get_tnt(),  # 'net':net.get_tnt() for tnt network // net.state_dict()
                    'acc': acc * 100.,
                    'epoch': epoch,
                }
                if not os.path.isdir('./saved/{}'.format(dict_name)):
                    os.makedirs('./saved/{}'.format(dict_name))
                torch.save(state, path)
                best_acc = acc
    return acc, test_loss, best_acc


def current_learning_rate(epoch, current_lr, args):
    if (epoch + 1) % args.d_epoch == 0:
        return current_lr * 0.1
    else:
        return current_lr


def ternary_convert(network):
    net_error = copy.deepcopy(network)  # create a network for saving local errors
    w = copy.deepcopy(network.state_dict())  # for local weights
    w_error = copy.deepcopy(net_error.state_dict())  # for local errors

    for name, module in network.named_modules():  # calculating errors between normal weights and errors
        if isinstance(module, TNTConv2d):
            #             print('convolution ternary')
            w[name + str('.weight')] = KernelsCluster.apply(module.weight)  # tnt
            w_error[name + str('.weight')] -= w[name + str('.weight')]  # errors

        if isinstance(module, TNTLinear):
            #             print('linear ternary')
            w[name + str('.weight')] = KernelsCluster.apply(module.weight)
            w_error[name + str('.weight')] -= w[name + str('.weight')]

    network.load_state_dict(w, strict=False)  # load tnt to model
    tnt_weights_dict = network.state_dict()  # creating a tensor form dict

    net_error.load_state_dict(w_error, strict=False)
    tnt_error_dict = net_error.state_dict()
    t = zero_rates(tnt_weights_dict)
    print('[INFO] zero rates is ', t)

    return tnt_weights_dict, tnt_error_dict


def float_pass(tnt, w, network):
    pass_w = copy.deepcopy(w)
    for name, module in network.named_modules():
        if isinstance(module, TNTConv2d):
            mask = (tnt[name + str('.weight')] == 0).float()
            pass_w[name + str('.weight')] = w[name + str('.weight')] * mask
    return pass_w


def FedAvg(w_dict, acc_dict, num):
    for i in range(num):
        lowest_acc_idx = min(acc_dict, key=acc_dict.get)
        w_dict.pop(lowest_acc_idx)
        acc_dict.pop(lowest_acc_idx)
    w = list(w_dict.values())

    w_avg = copy.deepcopy(w[0])
    # for k in w_avg.keys():
    #     w_count = (w_avg[k] != 0).float()
    #     for i in range(1, len(w)):
    #         w_avg[k] += w[i][k]
    #         mask = (w[i][k] != 0).float()
    #         w_count += mask
    #     w_avg[k] = torch.div(w_avg[k], w_count + 1e-7)

    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], float(len(w)))

    total_params = 0
    zero_params = 0
    for key in w_avg.keys():
        zero_params += (w_avg[key].view(-1) == 0).sum().item()
        total_params += len(w_avg[key].view(-1))

    return w_avg, (zero_params / total_params)


def rec_w(avg_tnt, local_err, net):
    # recover weights from avg_tnt and local_err
    for name, module in net.named_modules():
        if isinstance(module, TNTConv2d):
            avg_tnt[name + str('.weight')] += local_err[name + str('.weight')]
        if isinstance(module, TNTLinear):
            avg_tnt[name + str('.weight')] += local_err[name + str('.weight')]
    net.load_state_dict(avg_tnt)
    return net


def zero_rates(weight_dict):
    total_params = 0
    zero_params = 0
    for key in weight_dict.keys():
        #         print(key)
        zero_params += (weight_dict[key].view(-1) == 0).sum().item()
        total_params += len(weight_dict[key].view(-1))
    return (zero_params / total_params)


def store_weights(model):
    model_dict = {}
    for name, param in model.named_parameters():
        model_dict[name] = param.data.clone()
        # param.requires_grad = False
    return model_dict


def WeightsUpdate(global_new, global_old, omiga):
    old_weights = store_weights(global_old)
    new_weights = store_weights(global_new)

    for key in old_weights.keys():
        old_weights[key] += omiga * new_weights[key]

    for name, param in global_old.named_parameters():
        param.data.copy_(old_weights[name])
        # param.requires_grad = True
    # return global_old


def Client_net(net, num_users):
    cli_net = {}
    for i in range(num_users):
        cli_net[str(i)] = copy.deepcopy(net)
    return cli_net
