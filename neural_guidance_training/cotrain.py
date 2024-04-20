import argparse
import os
import random
import time
import warnings

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

from model import CoTrainNet, RegLoss, correlation
from utils import save_checkpoint, Summary, AverageMeter, ProgressMeter, accuracy, pickle_dump, make_directory
from Encoders import Encoders
from torch_gnet import Gnet
import data_loader

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch imagenet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenette2',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--save_dir', type=str, help='path to save the checkpoints')
parser.add_argument('--img_folder_txt', type=str, help='path to a textfile of image folders used')
parser.add_argument('--train_id', type=str, help='unique identifier')
parser.add_argument('--neural_predictor_pth', type=str, help='path to a neural predictor')

parser.add_argument('--roi', default='V1', type=str,
                    help='roi name: [V1], [hV4]...')
parser.add_argument('--neural_predictor_pos', default="layer4", type=str,
                    help='[layer1], [layer2], [layer3], [[layer4]]')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--neural-arch', default='resnet18',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--shuffle', action='store_true', help='We used shuffled image for the neural predictor')

parser.add_argument('--train_workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--test_workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--saved-data', action='store_true', help="use saved data")
parser.add_argument("--save-interval", default=2, type=int,
                    help='checkpointing interval')
parser.add_argument('--alpha', default=0.9, type=float,
                    help='regularization parameter')

best_acc1 = 0


def main():
    args = parser.parse_args()

    print("\n***check params ---------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("--------------------------\n")

    print("learning rate is", args.lr)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.shuffle:
        save_dir_name = f"{args.save_dir}/{args.arch}_neural_arch_{args.neural_arch}_trained_" \
                        f"{args.train_id}_roi_{args.roi}_shuffle_{args.neural_predictor_pos}"
    else:
        save_dir_name = f"{args.save_dir}/{args.arch}_neural_arch_{args.neural_arch}_trained_" \
                        f"{args.train_id}_roi_{args.roi}_{args.neural_predictor_pos}"

    print(f"*** Saving to: {save_dir_name}")

    if args.rank == 0:
        make_directory(args.save_dir)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        classifier = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        classifier = models.__dict__[args.arch]()

    if args.roi in ["random", "None"]:
        if args.roi == "None":
            args.alpha = 0.0
            print("!!!!!!!! alpha set to 0.0 when ROI is [None]")
        n_vox = 497
        neural_predictor = Encoders(args.neural_arch, n_vox).net
        print(f"\t--> Using {args.roi}-initialized {args.neural_arch} neural predictor with", end=' ')
    else:
        neural_predictor = torch.load(args.neural_predictor_pth)
        print(f"\t--> Using {args.roi}-trained {args.neural_arch} neural predictor with", end=' ')

    if type(neural_predictor) == torchvision.models.resnet.ResNet:
        num_voxels = neural_predictor.fc.out_features
    elif type(neural_predictor) == torchvision.models.AlexNet:
        num_voxels = neural_predictor.classifier[-1].out_features
    elif type(neural_predictor) == torchvision.models.EfficientNet:
        num_voxels = neural_predictor.classifier[-1].out_features
    elif type(neural_predictor) == Gnet:
        num_voxels = neural_predictor.subject_fwrf.nv
    print(f"\t--> number of voxels: {num_voxels}")

    model = CoTrainNet(classifier, neural_predictor, num_voxels, neural_head_pos=args.neural_predictor_pos)

    if args.rank == 0:
        print(model)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = RegLoss(alpha=args.alpha).to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1
                # best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # train_loader, val_loader, train_sampler, val_sampler = load_data(args)
    img_folder_ls = data_loader.load_img_folder_ls(args.img_folder_txt)

    ### data Loader
    train_loader, val_loader, train_sampler, val_sampler = \
        data_loader.load_data_folder(args.data, img_folder_ls, args.distributed, args.batch_size,
                                     args.train_workers, args.test_workers, shuffle=args.shuffle)

    if args.shuffle:
        print(f"Data loaded: train: {len(train_loader)}, val: {len(val_loader)}, SHUFFLE!!!", flush=True)
    else:
        print(f"Data loaded: train: {len(train_loader)}, val: {len(val_loader)}", flush=True)

    if args.evaluate:
        validate(val_loader, model, criterion, device, args)
        return

    train_loss_classify_per_epoch = []
    train_loss_regularize_per_epoch = []
    train_acc1_per_epoch = []
    train_acc5_per_epoch = []
    acc1_per_epoch = []
    acc5_per_epoch = []
    loss_classify_per_epoch = []
    loss_regularize_per_epoch = []
    corr_avg_per_epoch = []
    corr_max_per_epoch = []

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_loss_classify, train_loss_regularize, train_top1, train_top5 = train(
            train_loader, model, criterion, optimizer, epoch, device, args
        )
        # evaluate on validation set
        acc1, acc5, loss_classify, loss_regularize, corr_avg, corr_max = validate(
            val_loader, model, criterion, device, args
        )
        train_loss_classify_per_epoch.append(train_loss_classify)
        train_loss_regularize_per_epoch.append(train_loss_regularize)
        train_acc1_per_epoch.append(train_top1)
        train_acc5_per_epoch.append(train_top5)
        acc1_per_epoch.append(acc1)
        acc5_per_epoch.append(acc5)
        loss_classify_per_epoch.append(loss_classify)
        loss_regularize_per_epoch.append(loss_regularize)
        corr_avg_per_epoch.append(corr_avg)
        corr_max_per_epoch.append(corr_max)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        best_acc1 = max(acc1, best_acc1)

        if (
                epoch % args.save_interval == 0 and
                epoch != 0 and
                args.multiprocessing_distributed and
                args.rank % ngpus_per_node == 0
        ):
            state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(state, f"{save_dir_name}_epoch_{epoch}.pth")

    if (
            args.multiprocessing_distributed and
            args.rank % ngpus_per_node == 0
    ):
        state = {
            'epoch': args.epochs + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, f"{save_dir_name}_epoch_{args.epochs}_final.pth")
        pickle_dump({
            "train_loss_classify_per_epoch": train_loss_classify_per_epoch,
            "train_loss_regularize_per_epoch": train_loss_regularize_per_epoch,
            "train_acc1_per_epoch": train_acc1_per_epoch,
            "train_acc5_per_epoch": train_acc5_per_epoch,
            "acc1_per_epoch": acc1_per_epoch,
            "acc5_per_epoch": acc5_per_epoch,
            "loss_classify_per_epoch": loss_classify_per_epoch,
            "loss_regularize_per_epoch": loss_regularize_per_epoch,
            "corr_avg_per_epoch": corr_avg_per_epoch,
            "corr_max_per_epoch": corr_max_per_epoch,
        },
            f"{save_dir_name}.pkl"
        )


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_classify = AverageMeter('Loss_classify', ':.4e')
    losses_regularize = AverageMeter('Loss_regularize', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_classify, losses_regularize, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target, shuffled_img) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if args.shuffle:
            neural_predict, neural_out, output = model(images, shuffled_img)
        else:
            neural_predict, neural_out, output = model(images)

        loss_classify, loss_regularize = criterion(neural_predict, neural_out, output, target)
        loss = loss_classify + loss_regularize

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses_classify.update(loss_classify.item(), images.size(0))
        losses_regularize.update(loss_regularize.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.rank == 0:
            progress.display(i + 1)
            print("", flush=True)

    return losses_classify.avg, losses_regularize.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, device, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target, _) in enumerate(loader):
                i = base_progress + i
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # compute output
                neural_predict, neural_out, output = model(images)
                loss_classify, loss_regularize = criterion(neural_predict, neural_out, output, target)
                corr = correlation(neural_predict, neural_out)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses_classify.update(loss_classify.item(), images.size(0))
                losses_regularize.update(loss_regularize.item(), images.size(0))
                correlation_avg.update(torch.mean(corr).item(), images.size(0))
                correlation_max.update(torch.max(corr).item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses_classify = AverageMeter('Loss_classify', Summary.NONE)
    losses_regularize = AverageMeter('Loss_regularize', Summary.NONE)
    correlation_avg = AverageMeter('Correlation_avg', Summary.AVERAGE)
    correlation_max = AverageMeter('Correlation_max', Summary.MAX)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses_classify, losses_regularize, correlation_avg, correlation_max, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()
        losses_classify.all_reduce()
        losses_regularize.all_reduce()
        correlation_avg.all_reduce()
        correlation_max.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.test_workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    if args.rank == 0:
        progress.display_summary()

    return top1.avg, top5.avg, losses_classify.avg, losses_regularize.avg, correlation_avg.avg, correlation_max.avg


if __name__ == '__main__':
    main()
