import torch
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock
import torch.nn as nn

class Encoders():
    def __init__(self, model, num_voxel):
        torch.manual_seed(1024)
        if model == "alexnet":
            from torchvision.models import alexnet
            self.net = alexnet(num_classes=num_voxel)
        elif model == "resnet18":
            from torchvision.models import resnet18
            self.net = resnet18(num_classes=num_voxel)
        elif model == "resnet50":
            from torchvision.models import resnet50
            self.net = resnet50(num_classes=num_voxel)
        elif model == "gnet":
            from torch_gnet import Gnet
            self.net = Gnet(num_voxel=num_voxel)
        elif model == "squeezenet":
            from torchvision.models import SqueezeNet
            self.net = SqueezeNet(num_classes=num_voxel)
        elif model == "efficientnet_b0":
            from torchvision.models import efficientnet_b0
            self.net = efficientnet_b0(num_classes=num_voxel)
        elif model == "vgg16":
            from torchvision.models import vgg16
            self.net = vgg16(num_classes=num_voxel)
        else:
            raise

