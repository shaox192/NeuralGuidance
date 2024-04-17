import os
import torch
import torch.distributed as dist
import pickle as pkl

from torchvision import models
# from Encoders import Encoders
# from model import CoTrainNet, AttackNet

def make_directory(pth):
    if not os.path.exists(pth):
        print(f"Making output dir at {pth}")
        os.makedirs(pth, exist_ok=True)
    else:
        print(f"Path {pth} exists.")


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


# def instantiate_ROI_model(model_f, device, victim="classification"):
#     checkpoint = torch.load(model_f, map_location=device)
#     num_voxels = checkpoint["state_dict"]['module.neural_predictor.fc.weight'].size()[0]
#     neural_predictor = models.resnet18(num_classes=num_voxels)
#
#     classifier = models.resnet18(pretrained=True)
#
#     # instantiate a model (could also be a TensorFlow or JAX model)
#     cotrain_model = CoTrainNet(classifier, neural_predictor, num_voxels, neural_head_pos=pos)
#     model = AttackNet(cotrain_model, victim=victim)
#
#     model.load_state_dict(checkpoint['state_dict'])
#
#     return model


def load_from_nii(mask_nii_file):
    import nibabel as nib
    if os.path.exists(mask_nii_file):
        return nib.load(mask_nii_file).get_fdata()
    else:
        raise FileNotFoundError(f"can't find mask file: {mask_nii_file}!")



def pickle_dump(data, fpth):
    print(f"writing to: {fpth}")
    with open(fpth, 'wb') as f:
        pkl.dump(data, f)


def pickle_load(fpth):
    print(f"loading from: {fpth}")
    with open(fpth, 'rb') as f:
        return pkl.load(f)


def show_input_args(args):
    print("\n***check params ---------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("--------------------------\n", flush=True)
