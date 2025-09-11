from enum import Enum
import os
import torch
import shutil
import torch.distributed as dist
import pickle as pkl
import numpy as np

from torchvision import models

import torch.distributed as dist


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
    MAX = 4

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE, loss_alpha=1):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.loss_alpha = loss_alpha
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)

        max_ = torch.tensor([self.max], dtype=torch.float32, device=device)
        dist.all_reduce(max_, dist.ReduceOp.MAX, async_op=False)

        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count
        self.max = max_

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        if self.loss_alpha != 1:
            fmtstr += f" ({self.loss_alpha: .2f} weighted loss {self.loss_alpha * self.val :.4f}):"
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        elif self.summary_type is Summary.MAX:
            fmtstr = '{name} {max:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [f" *{self.prefix}* summary:"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = min(max(topk), output.shape[-1])  # number of classes, just in case of less classes than 5 during debugging
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




def load_np_NG(ng_pth, device):
    # these are the NG models trained from BIAI time, so there's module before the state_dict
    checkpoint = torch.load(ng_pth, map_location=device)
    num_voxels = checkpoint["state_dict"]['module.neural_predictor.fc.weight'].size()[0]
    np_state_dict = {k[len("module.neural_predictor."): ]: v for k, v in checkpoint["state_dict"].items() if "neural_predictor" in k}

    neural_predictor = models.resnet18(num_classes=num_voxels)
    neural_predictor.load_state_dict(np_state_dict)
    return neural_predictor


def pickle_dump(data, fpth):
    print_safe(f"writing to: {fpth}")
    # if is_main_process():
    with open(fpth, 'wb') as f:
        pkl.dump(data, f)


def pickle_load(fpth):
    print_safe(f"loading from: {fpth}")
    # if is_main_process():
    with open(fpth, 'rb') as f:
        return pkl.load(f)

def pickle_append(data, fpth, new_fpth):
    print("Appending......")
    with open(fpth, 'rb') as f:
        data_dict = pkl.load(f)
    
    for k, v in data.items():
        data_dict[k] = v
    pickle_dump(data_dict, new_fpth)

def show_input_args(args):
    print_safe("\n***check params ---------")
    for arg in vars(args):
        print_safe(f"{arg}: {getattr(args, arg)}")
    print_safe("--------------------------\n")



def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def print_safe(print_str, **kwargs):
    if "flush" not in kwargs:
        kwargs["flush"] = True
    if is_main_process():
        print(print_str, **kwargs)

def make_directory(pth):
    if is_main_process():
        if not os.path.exists(pth):
            print(f"Making output dir at {pth}")
            os.makedirs(pth, exist_ok=True)
        else:
            print(f"Path {pth} exists.")



def calc_radius(data_points, center_norm_vals):
    ds0 = data_points - data_points.mean(axis=1, keepdims=True)  # mean centering

    ds = ds0/center_norm_vals  # normalized by the distance to the center, thus in "center norm unit" as in the paper
    ds_sq_sum = np.sum(np.square(ds), axis=0)

    R_M = np.sqrt(np.mean(ds_sq_sum))
    # not the same as : rr = np.linalg.norm(ds0, axis=0).mean(), they have RMS
    return R_M


def calc_svd_dim(data_mat):
    U, S, Vh = np.linalg.svd(data_mat)
    var_explained = np.cumsum(S**2 / np.sum(S**2))
    return U, S, var_explained