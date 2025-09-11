"""
Cotrain a neural network with specific manifold statistics
"""


import argparse
import os
import random
import time
import warnings
import numpy as np
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import Subset

from man_models import ManCoTrainNet, ManLoss
import man_utils
import man_data_loader

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch imagenet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenette2',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--img_folder_txt', type=str, help='path to a textfile of image folders used')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

######### original space tuned NP feats
parser.add_argument('--orig-np-feats', type=str, help='path to the pkl with original space neural predictor feats')
# parser.add_argument('--orig-imagenet-lbs', type=str, help='path to the orig imagenet mapping')
parser.add_argument('--save-man-stats', type=str, default=None, help='save the computed svd stats for future uses')
parser.add_argument('--precomputed-man-stats', type=str, default=None, help='path to the precomputed manifold stats')

######### run with MFTMA manifold stats, so decorr on
parser.add_argument('--decorr', action='store_true', help='enable calculation of mftma for decorrelated space')
parser.add_argument('--manifold-stats', type=str, help='path to the pkl with estimated manifold stats')

######### NNR params  *** if input these, will override the manifold stats
parser.add_argument('--NNR-dim', action="store_true", help='whether to turn on soft NNR regularization for each category manifold')
parser.add_argument('--NNR-rad', type=float, help='radius, will be applied to all categories')

######### reg loss alphas
parser.add_argument('--alphas', type=float, nargs='*', default=[0.7, 0.15, 0.15, 0.0, 0.0],
                    help='regularization parameter')


########## subj/roi parameter
parser.add_argument('--roi', default='V1', type=str,
                    help='roi name: [V1], [hV4]...')
parser.add_argument('--sub', type=str, help='subject id')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')


########## other parameters
# parser.add_argument('--shuffle', action='store_true', help='We used shuffled image for the neural predictor')
# no shuffle anymore
parser.add_argument('--train_workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--test_workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
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

#### save stuff
parser.add_argument('--saved-data', action='store_true', help="use saved data")
parser.add_argument('--NO-SAVE', action='store_true', help="turn this on to not save anything")
parser.add_argument('--save-dir', type=str, help='path to save the checkpoints')
parser.add_argument('--save-ID', type=str, default=None, help='any special id appended to the save directory')
parser.add_argument("--save-interval", default=2, type=int,
                    help='checkpointing interval')

#### scheduler
parser.add_argument('--scheduler-gamma', type=float, default=0.1, help='stepLR scheduler gamma')
parser.add_argument('--scheduler-step-size', type=int, default=20, help='stepLR scheduler reduce step size')



def get_model_weights(arch):
    if arch == "resnet18":
        return models.ResNet18_Weights.DEFAULT
    elif arch == "resnet50":
        return models.ResNet50_Weights.DEFAULT
    else:
        raise NotImplementedError


def build_data_loader(args):
    img_folder_ls = man_data_loader.load_img_folder_ls(args.img_folder_txt)

    ### data Loader
    train_loader, val_loader, train_sampler, val_sampler = \
        man_data_loader.load_data_folder(
            args.data, 
            img_folder_ls, 
            args.distributed, 
            args.batch_size,
            args.train_workers, 
            args.test_workers, 
        )

    print(f"Data loaded: train: {len(train_loader)}, val: {len(val_loader)}", flush=True)
    
    return train_loader, val_loader, train_sampler, val_sampler


def main():
    args = parser.parse_args()

    if len(args.alphas) != 3:
        raise ValueError("Max number of reegularization alphas should be a list of 3 floats")

    if args.decorr:
        raise NotImplementedError("Decorrelation not implemented yet")
    if args.roi == "None":
        args.alphas = (1.0, 0.0, 0.0)
        man_utils.print_safe("!!!!!!!! alpha set to 0.0 when ROI is [None]")

    args.alphas = tuple(args.alphas) # [a/sum(args.alphas) for a in args.alphas])

    man_utils.print_safe("\n***check params ---------")
    for arg in vars(args):
        man_utils.print_safe(f"{arg}: {getattr(args, arg)}")
    man_utils.print_safe("--------------------------\n")

    man_utils.print_safe(f"* learning rate is {args.lr}")

    ## -- set up save directory
    if args.NO_SAVE:
        save_dir_name=None
        man_utils.print_safe("!!!!!!!!!!NO SAVE MODE!!!!!!")
    else:
        datetime_str = (
        str(datetime.datetime.now().replace(microsecond=0))
        .replace(" ", "-")
        .replace(":", "-")
        )
        save_dir_name = f"{args.save_dir}/{args.sub}_roi_{args.roi}_coco50_fc50-{datetime_str}"
        man_utils.make_directory(save_dir_name)
        man_utils.print_safe(f"*** Saving to: {save_dir_name}")
    # -- 

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        args.rank = 0
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, save_dir_name))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, save_dir_name)


def main_worker(gpu, ngpus_per_node, args, save_dir_name=None):

    ## -- set up distributed training
    args.gpu = gpu

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # if you use multiple GPUs
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn(f'SEEDING training: seed {args.seed}')

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

    ## -- find device
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ## -- load data
    #### **** check batch size
    if args.distributed and torch.cuda.is_available() and (args.gpu is not None):
        args.batch_size = int(args.batch_size / ngpus_per_node)
        man_utils.print_safe(f"! Batch size adjusted to {args.batch_size} for distributed training")

    train_loader, val_loader, train_sampler, val_sampler = build_data_loader(args)

    ## -- load manifold stats
    if args.roi == "None":
        # np_g = np.random.default_rng(seed=1024) # for debugging purposes.
        num_cls = len(train_loader.dataset.class_to_idx)
        man_stats = {"basis": torch.tensor(np.stack([np.eye(497) for _ in range(num_cls)], 
                                                    axis=0),
                                           dtype=torch.float32), 
                     "center": torch.tensor(np.vstack([np.zeros(497) for _ in range(num_cls)]), 
                                            dtype=torch.float32), 
                     "rad": torch.tensor(np.ones(num_cls),
                                            dtype=torch.float32)
                    }
    elif args.precomputed_man_stats:
        man_utils.print_safe(f"Loading precomputed manifold stats from {args.precomputed_man_stats}")
        man_stats = man_utils.pickle_load(args.precomputed_man_stats)
    else:
        man_stats = man_data_loader.get_man_stats_from_np_feats(args.orig_np_feats, train_loader.dataset.class_to_idx)
        if args.save_man_stats:
            man_utils.pickle_dump(man_stats, args.save_man_stats)
            man_utils.print_safe(f"Manifold stats saved to {args.save_man_stats}, exiting...")
            exit()

    ## -- create model
    if args.pretrained:
        man_utils.print_safe("=> using pre-trained model '{}'".format(args.arch))
        classifier = models.__dict__[args.arch](weights=get_model_weights(args.arch))
    else:
        man_utils.print_safe("=> creating model '{}'".format(args.arch))
        classifier = models.__dict__[args.arch]()

    if args.roi == "None":
        n_vox = 497
    else:
        n_vox = man_stats["basis"][0].shape[0]

    man_utils.print_safe(f"\t--> number of voxels for {args.roi}: {n_vox}")

    model = ManCoTrainNet(classifier, n_vox, num_classes=len(train_loader.dataset.class_to_idx))
    man_utils.print_safe(model)

    if not torch.cuda.is_available():
        man_utils.print_safe('using CPU, this will be slow')
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
                # args.batch_size = int(args.batch_size / ngpus_per_node)
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


    # print("ON GPU: {}, model cls {}; model np {}".format(args.gpu, model.shared_layer.conv1.weight[20, 1, 2, :4], 
    #                                                      model.neural_head.neural_fc.weight[20, :4]))

    if args.gpu is not None:
        print(f"CHECK RANDOM SEEDING: ON GPU: {args.gpu}\n"
              f"shared model {model.module.shared_layer.conv1.weight[20, 1, 2, :3]}, \n"
              f"NP FC weights {model.module.neural_head.neural_fc.weight[20, :3]}")

    # define loss function (criterion), optimizer, and learning rate scheduler
    if args.NNR_dim:
        man_utils.print_safe("* -- Using soft NNR regularization")
    if args.NNR_rad:
        man_utils.print_safe(f"* -- Using NNR radius regularization with radius {args.NNR_rad}")
    
    criterion = ManLoss(man_stats, 
                        args.NNR_dim, args.NNR_rad, 
                        man_utils.print_safe, 
                        args.gpu,
                        abl=args.abl, abl_num_dim=args.abl_num_dim,
                        decorr=args.decorr).to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                              weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    # scheduler = CosineAnnealingLR(optimizer, T_max=100)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            man_utils.print_safe("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']

            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            man_utils.print_safe("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            man_utils.print_safe("=> no checkpoint found at '{}'".format(args.resume))


    if args.evaluate:
        validate(val_loader, model, criterion, device, args, man_stats["man_info_retriev_idx"])
        return

    train_loss_cls_epk = []
    train_loss_reg_rad_epk = []
    train_loss_reg_dim_epk = []
    train_acc1_epk = []
    train_acc5_epk = []

    val_loss_cls_epk = []
    val_loss_reg_rad_epk = []
    val_loss_reg_dim_epk = []
    val_acc1_epk = []
    val_acc5_epk = []

    man_utils.print_safe(f"Starting training from epk-{args.start_epoch} to epk-{args.epochs}")
    for epoch in range(args.start_epoch, args.epochs):
        man_utils.print_safe(f"Epoch: {epoch}, scheduler lr: {scheduler.get_last_lr()}, opt lr: {optimizer.param_groups[0]['lr']}")
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_loss_classify, train_loss_rad, train_loss_dim, \
        train_top1, train_top5 = train(train_loader, model, criterion, optimizer, \
                                           epoch, device, args) #man_stats["man_info_retriev_idx"])

        # evaluate on validation set
        loss_classify, val_loss_rad, val_loss_dim, \
            val_top1, val_top5 = validate(
            val_loader, model, criterion, device, args  # , man_stats["man_info_retriev_idx"]
        )


        train_acc1_epk.append(train_top1)
        train_acc5_epk.append(train_top5)
        train_loss_cls_epk.append(train_loss_classify)
        train_loss_reg_rad_epk.append(train_loss_rad)
        train_loss_reg_dim_epk.append(train_loss_dim)

        val_acc1_epk.append(val_top1)
        val_acc5_epk.append(val_top5)
        val_loss_cls_epk.append(loss_classify)
        val_loss_reg_rad_epk.append(val_loss_rad)
        val_loss_reg_dim_epk.append(val_loss_dim)

        scheduler.step()

        if (    args.NO_SAVE is False and
                epoch % args.save_interval == 0 and
                epoch != 0 and
                man_utils.is_main_process()
        ):
            state = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.module.state_dict() if args.multiprocessing_distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(state, f"{save_dir_name}/epk_{epoch}.pth")

            man_utils.pickle_dump({
                "train_loss_cls_epk": train_loss_cls_epk,
                "train_loss_reg_rad_epk": train_loss_reg_rad_epk,
                "train_loss_reg_dim_epk": train_loss_reg_dim_epk,
                "train_acc1_epk": train_acc1_epk,
                "train_acc5_epk": train_acc5_epk,

                "val_loss_cls_epk": val_loss_cls_epk,
                "val_loss_reg_rad_epk": val_loss_reg_rad_epk,
                "val_loss_reg_dim_epk": val_loss_reg_dim_epk,
                "val_acc1_epk": val_acc1_epk,
                "val_acc5_epk": val_acc5_epk,
                }, 
                f"{save_dir_name}/stats.pkl")


def train(train_loader, model, criterion, optimizer, epoch, device, args): # , man_info_retriev_idx):
    # switch to train mode
    model.train()

    batch_time = man_utils.AverageMeter('Time', ':6.3f')
    data_time = man_utils.AverageMeter('Data', ':6.3f')
    losses_classify = man_utils.AverageMeter('Loss_classify', ':.4e', loss_alpha=args.alphas[0])
    losses_reg_orig_rad = man_utils.AverageMeter('Loss_reg_orig_rad', ':.4e', loss_alpha=args.alphas[1])
    losses_reg_orig_dim = man_utils.AverageMeter('Loss_reg_orig_dim', ':.4e', loss_alpha=args.alphas[2])
    top1 = man_utils.AverageMeter('Acc@1', ':6.2f')
    top5 = man_utils.AverageMeter('Acc@5', ':6.2f')
    progress = man_utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, 
         losses_classify, losses_reg_orig_rad, losses_reg_orig_dim,
         top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, (images, target, _) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # obtain the corresponding index of the manifold data
        # index2select = [man_info_retriev_idx[t] for t in target]

        # move data to the same device as model
        # index2select = torch.tensor(index2select, dtype=int).to(device)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # compute output
        neural_out, classification_out = model(images)

        loss_classify, loss_orig_rad, loss_orig_dim = criterion(classification_out, 
                                                                target, 
                                                                neural_out, 
                                                                device=device)
        
        loss = args.alphas[0] * loss_classify + \
               args.alphas[1] * loss_orig_rad + \
               args.alphas[2] * loss_orig_dim
        
        # if args.gpu is not None and i < 5:
        #     print("epk={}, i={}, ON GPU: {}, loss {}".format(epoch, i, args.gpu, loss))

        # measure accuracy
        acc1, acc5 = man_utils.accuracy(classification_out, target, topk=(1, 5))

        # record progress
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        losses_classify.update(loss_classify.item(), images.size(0))
        losses_reg_orig_rad.update(loss_orig_rad.item(), images.size(0))
        losses_reg_orig_dim.update(loss_orig_dim.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0 and man_utils.is_main_process():
            progress.display(i + 1)
            man_utils.print_safe("")

    return losses_classify.avg, losses_reg_orig_rad.avg, losses_reg_orig_dim.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, device, args): # , man_info_retriev_idx):
    # switch to evaluate mode
    model.eval()

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target, _) in enumerate(loader):
                i = base_progress + i

                # # obtain the corresponding index of the manifold data
                # index2select = [man_info_retriev_idx[t] for t in target]

                # move data to the same device as model
                # index2select = torch.tensor(index2select, dtype=int).to(device)
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # compute output
                neural_out, classification_out = model(images)

                loss_classify, loss_orig_rad, loss_orig_dim = criterion(classification_out, 
                                                                target, 
                                                                neural_out, 
                                                                device=device)
                
                # loss_classify *= args.alphas[0]
                # loss_orig_space *= args.alphas[1]
                # loss_decorr_rad *= args.alphas[2]
                # loss_decorr_dim *= args.alphas[3]
                
                # measure accuracy and record loss
                acc1, acc5 = man_utils.accuracy(classification_out, target, topk=(1, 5))
                losses_classify.update(loss_classify.item(), images.size(0))

                losses_reg_orig_rad.update(loss_orig_rad.item(), images.size(0))
                losses_reg_orig_dim.update(loss_orig_dim.item(), images.size(0))

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

    batch_time = man_utils.AverageMeter('Time', ':6.3f', man_utils.Summary.NONE)
    losses_classify = man_utils.AverageMeter('Loss_classify', man_utils.Summary.NONE, loss_alpha=args.alphas[0])

    losses_reg_orig_rad = man_utils.AverageMeter('Loss_reg_orig_rad', ':.4e', loss_alpha=args.alphas[1])
    losses_reg_orig_dim = man_utils.AverageMeter('Loss_reg_orig_dim', ':.4e', loss_alpha=args.alphas[2])

    top1 = man_utils.AverageMeter('Acc@1', ':6.2f', man_utils.Summary.AVERAGE)
    top5 = man_utils.AverageMeter('Acc@5', ':6.2f', man_utils.Summary.AVERAGE)

    progress = man_utils.ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, 
         losses_classify, losses_reg_orig_rad, losses_reg_orig_dim, 
         top1, top5],
        prefix='Test: ')

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()
        losses_classify.all_reduce()

        losses_reg_orig_rad.all_reduce()
        losses_reg_orig_dim.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.test_workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    if man_utils.is_main_process():
        progress.display_summary()

    return losses_classify.avg, losses_reg_orig_rad.avg, losses_reg_orig_dim.avg, \
           top1.avg, top5.avg


if __name__ == '__main__':
    main()