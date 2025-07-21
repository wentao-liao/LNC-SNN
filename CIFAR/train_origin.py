import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import math
from torch.cuda import amp
import torch.distributed.optim
import argparse
from spikingjelly.clock_driven import functional
import utils
from models import ms_resnet
import matplotlib.pyplot as plt
# _seed_ = 1000
# import random
# random.seed(1000)
# torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
# torch.cuda.manual_seed_all(_seed_)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

import numpy as np
# np.random.seed(_seed_)
import psutil
import GPUtil


def get_memory_info():
    """获取CPU和GPU内存使用情况"""

    # CPU内存信息

    process = psutil.Process(os.getpid())

    cpu_mem = process.memory_info().rss / 1024 / 1024  # 转换为MB

    # GPU内存信息

    gpu_mem = None

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # 转换为MB
        gpu_mem_reserved = torch.cuda.memory_reserved() / 1024 / 1024
    return {
        'cpu_mem_used': cpu_mem,
        'gpu_mem_used': gpu_mem,
        'gpu_mem_reserved': gpu_mem_reserved if gpu_mem is not None else None
    }


def print_memory_usage():
    """打印内存使用情况"""
    mem_info = get_memory_info()
    print(f"\nMemory Usage:")
    print(f"CPU Memory: {mem_info['cpu_mem_used']:.2f} MB")
    if mem_info['gpu_mem_used'] is not None:
        print(f"GPU Memory Used: {mem_info['gpu_mem_used']:.2f} MB")
        print(f"GPU Memory Reserved: {mem_info['gpu_mem_reserved']:.2f} MB")
        # 获取GPU详细信息
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id} - {gpu.name}:")
            print(f"    Total Memory: {gpu.memoryTotal} MB")
            print(f"    Free Memory: {gpu.memoryFree} MB")
            print(f"    Used Memory: {gpu.memoryUsed} MB")
            print(f"    GPU Load: {gpu.load * 100:.1f}%")


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        # with torch.autograd.detect_anomaly():
        if scaler is not None:
            with amp.autocast():
                # output = model(image)
                outputs = model(image)
                loss = criterion(outputs, target)
        else:
            # output = model(image)
            outputs = model(image)
            loss = criterion(outputs, target)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        functional.reset_net(model)
        acc1, acc5 = utils.accuracy(outputs, target, topk=(1, 5))
        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()
        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # output = model(image)
            outputs = model(image)
            loss = criterion(outputs, target)
            functional.reset_net(model)
            acc1, acc5 = utils.accuracy(outputs, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes

    metric_logger.synchronize_between_processes()
    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    print_memory_usage()
    return loss, acc1, acc5


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, cache_dataset, distributed):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)
    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)
    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.
    train_tb_writer = None
    te_tb_writer = None
    utils.init_distributed_mode(args)
    print(args)
    output_dir = os.path.join(args.output_dir, f'{args.model}_b{args.batch_size}_lr{args.lr}_T{args.T}')
    if args.zero_init_residual:
        output_dir += '_zi'
    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'
    output_dir += '_steplr'
    if args.adam:
        output_dir += '_adam'
    else:
        output_dir += '_sgd'
    if output_dir:
        utils.mkdir(output_dir)
    device = torch.device(args.device)
    batch_size = args.batch_size
    dataset_root_dir = args.data_path
    data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            transform=torchvision.transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.557, 0.549, 0.5534])

            ]),

            download=True),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=args.workers)
    data_loader_test = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            transform=torchvision.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.557, 0.549, 0.5534])
            ]),
            download=True),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.workers)
    print("Creating model")
    if args.model in ms_resnet.__dict__:
        model = ms_resnet.__dict__[args.model](T=args.T)
    else:
        raise NotImplementedError(args.model)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    criterion = nn.CrossEntropyLoss()
    if args.adam:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cos_lr_T)
    # 学习率更新
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120,190], gamma=0.1)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']
    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        return

    # if args.tb and utils.is_main_process():

    if utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))
        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')
    print("Start training")

    start_time = time.time()
    train_accs = []
    # test_losses = []
    test_accs = []
    for epoch in range(args.start_epoch, args.epochs):
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc1, train_acc5 = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch,
                                                             args.print_freq, scaler)
        if utils.is_main_process():
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
        lr_scheduler.step()
        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        if te_tb_writer is not None:
            if utils.is_main_process():
                te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)
        plt.figure(figsize=(6, 4))
        plt.plot(train_accs, label='Train Acc', color='blue')  # 显式设置颜色
        plt.plot(test_accs, label='Test Acc', color='orange')
        plt.xlim(0, epoch)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curves')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
        if (epoch + 1) % 50 == 0:
            plt.show()
        plt.close()
        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True

        if output_dir:

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'max_test_acc1': max_test_acc1,
                'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,

            }

            utils.save_on_master(
                checkpoint,
                os.path.join(output_dir, 'checkpoint_latest.pth'))

            save_flag = False
            if epoch % 64 == 0 or epoch == args.epochs - 1:
                save_flag = True
            elif args.cos_lr_T == 0:
                for item in args.lr_step_size:
                    if (epoch + 2) % item == 0:
                        save_flag = True
                        break

            if save_flag:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

            if save_max:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))

        print(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(output_dir)

        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1,
              'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--data-path', default='../data', help='dataset')
    parser.add_argument('--model', default='ms_resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=1e-1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum for SGD. Adam will not use momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./output/logs', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",

    )

    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",

    )

    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",

    )

    # Mixed precision training parameters

    parser.add_argument('--amp', action='store_true',
                        help='Use AMP training')
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--tb', action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--adam', action='store_true',
                        help='Use Adam. The default optimizer is Adam.')
    parser.add_argument('--cos_lr_T', default=320, type=int,
                        help='T_max of CosineAnnealingLR.')
    parser.add_argument('--zero_init_residual', action='store_true', help='zero init all residual blocks')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)


