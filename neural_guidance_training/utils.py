from enum import Enum
import os
import torch
import shutil
import torch.distributed as dist
import pickle as pkl

from torchvision import models
from Encoders import Encoders
from model import CoTrainNet, AttackNet



def make_directory(pth):
    if not os.path.exists(pth):
        print(f"Making output dir at {pth}")
        os.makedirs(pth, exist_ok=True)
    else:
        print(f"Path {pth} exists.")



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
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def instantiate_ROI_model(model_f, pos, neural_predictor_arch, classifier_arch:str, device, victim="classification"):
    checkpoint = torch.load(model_f, map_location=device)

    if neural_predictor_arch == "resnet18":
        num_voxels = checkpoint["state_dict"]['module.neural_predictor.fc.weight'].size()[0]
        neural_predictor = models.resnet18(num_classes=num_voxels)
    else:
        raise NotImplementedError(f"neural_predictor_arch {neural_predictor_arch} not supported")

    classifier = models.__dict__[classifier_arch](pretrained=True)

    cotrain_model = CoTrainNet(classifier, neural_predictor, num_voxels, neural_head_pos=pos)
    model = AttackNet(cotrain_model, victim=victim)

    model.load_state_dict(checkpoint['state_dict'])

    return model



def pickle_dump(data, fpth):
    print(f"writing to: {fpth}")
    with open(fpth, 'wb') as f:
        pkl.dump(data, f)


def pickle_load(fpth):
    print(f"loading from: {fpth}")
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
    print("\n***check params ---------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("--------------------------\n")