import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict


class CoTrainNetLegacy(nn.Module):
    def __init__(self, classifier, neural_predictor, num_voxels, USE_HEAD=0):
        super(CoTrainNetLegacy, self).__init__()
        if type(classifier) is torchvision.models.resnet.ResNet:
            self.shared_layer = nn.Sequential(OrderedDict([
                ('conv1', classifier.conv1),
                ('bn1', classifier.bn1),
                ('relu', classifier.relu),
                ('maxpool', classifier.maxpool),
                ('layer1', classifier.layer1),
                ('layer2', classifier.layer2),
                ('layer3', classifier.layer3),
                ('layer4', classifier.layer4),
                ('avgpool', classifier.avgpool),
            ]))
            self.classification_head = classifier.fc
            self.neural_head = nn.Linear(classifier.fc.in_features, num_voxels)
        else:
            assert NotImplementedError
        self.neural_predictor = neural_predictor
        self.use_head = USE_HEAD
        for param in self.neural_predictor.parameters():
            param.requires_grad = False

    def forward(self, x, shuffled_img=None, classify_only=False):
        out = self.shared_layer(x)
        out = torch.flatten(out, 1)
        classification_out = self.classification_head(out)
        if classify_only:
            return classification_out
        neural_out = self.neural_head(out)
        if shuffled_img is not None:
            neural_predict = self.neural_predictor(shuffled_img)
        else:
            neural_predict = self.neural_predictor(x)
        return neural_predict, neural_out, classification_out


class CoTrainNet(nn.Module):
    def __init__(self, classifier, neural_predictor, num_voxels, num_classes=50, neural_head_pos="layer4", USE_HEAD=-1):
        """

        :param classifier:
        :param neural_predictor:
        :param num_voxels:
        :param neural_head_pos: "layer1", "layer2", "layer3", "layer4"
        :param USE_HEAD:
        """
        super(CoTrainNet, self).__init__()
        if type(classifier) is torchvision.models.resnet.ResNet:
            self.shared_layer = nn.Sequential(OrderedDict([
                ('conv1', classifier.conv1),
                ('bn1', classifier.bn1),
                ('relu', classifier.relu),
                ('maxpool', classifier.maxpool),
                ('layer1', classifier.layer1)
            ]))

            if neural_head_pos == "layer1":
                self.neural_head = nn.Sequential(OrderedDict([
                    ("neural_avgpool", nn.AdaptiveAvgPool2d(output_size=(4, 4))),
                    ("neural_flatten", nn.Flatten()),
                    ("neural_fc", nn.Linear(64 * 4 * 4, num_voxels)),
                    ]))

                self.classification_branch = nn.Sequential(OrderedDict([
                    ('layer2', classifier.layer2),
                    ('layer3', classifier.layer3),
                    ('layer4', classifier.layer4),
                    ('avgpool', classifier.avgpool),
                    # ('fc', classifier.fc)
                ]))
            elif neural_head_pos == "layer2":
                self.neural_head = nn.Sequential(OrderedDict([
                    ("neural_avgpool", nn.AdaptiveAvgPool2d(output_size=(2, 2))),
                    ("neural_flatten", nn.Flatten()),
                    ("neural_fc", nn.Linear(128 * 2 * 2, num_voxels)),
                    ]))

                self.shared_layer += nn.Sequential(OrderedDict([
                    ('layer2', classifier.layer2)
                ]))
                self.classification_branch = nn.Sequential(OrderedDict([
                    ('layer3', classifier.layer3),
                    ('layer4', classifier.layer4),
                    ('avgpool', classifier.avgpool),
                    # ('fc', classifier.fc)
                ]))
            elif neural_head_pos == 'layer3':
                self.neural_head = nn.Sequential(OrderedDict([
                    ("neural_avgpool", nn.AdaptiveAvgPool2d(output_size=(2, 2))),
                    ("neural_flatten", nn.Flatten()),
                    ("neural_fc", nn.Linear(256 * 2 * 2, num_voxels)),
                ]))

                self.shared_layer += nn.Sequential(OrderedDict([
                    ('layer2', classifier.layer2),
                    ('layer3', classifier.layer3)
                ]))
                self.classification_branch = nn.Sequential(OrderedDict([
                    ('layer4', classifier.layer4),
                    ('avgpool', classifier.avgpool),
                ]))
            else:
                self.neural_head = nn.Sequential(OrderedDict([
                    ("neural_flatten", nn.Flatten()),
                    ("neural_fc", nn.Linear(512, num_voxels)),
                ]))

                self.shared_layer += nn.Sequential(OrderedDict([
                    ('layer2', classifier.layer2),
                    ('layer3', classifier.layer3),
                    ('layer4', classifier.layer4),
                    ('avgpool', classifier.avgpool)
                ]))
                self.classification_branch = None

            # self.classification_head = classifier.fc
            self.classification_head = nn.Linear(512 * 1, num_classes)

        elif type(classifier) is torchvision.models.AlexNet:
            alexnet_classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
            )

            self.shared_layer = nn.Sequential(OrderedDict([
                ('features', classifier.features),
                ('avgpool', classifier.avgpool),
                ('flatten', nn.Flatten()),
                ('classifier', alexnet_classifier),
            ]))

            self.classification_branch = None
            self.neural_head = nn.Sequential(OrderedDict([
                ("neural_flatten", nn.Flatten()),
                ("neural_fc", nn.Linear(4096, num_voxels)),
            ]))
            self.classification_head = nn.Linear(4096, num_voxels)

        else:
            assert NotImplementedError

        self.neural_head_pos = neural_head_pos
        self.num_voxels = num_voxels

        self.use_head = USE_HEAD  # 8 heads each for 1 subject

        self.neural_predictor = neural_predictor
        for param in self.neural_predictor.parameters():
            param.requires_grad = False

    def _classification_head(self, x):
        classification_out = self.classification_branch(x) if self.classification_branch is not None \
            else torch.clone(x)
        classification_out = torch.flatten(classification_out, 1)
        classification_final = self.classification_head(classification_out)
        return classification_final

    def forward(self, x, shuffled_img=None, which_head=None):
        shared_out = self.shared_layer(x)

        if which_head == "classification":
            return self._classification_head(shared_out)

        elif which_head == "neural":
            return self.neural_head(shared_out)

        else:
            classification_final = self._classification_head(shared_out)
            neural_final = self.neural_head(shared_out)

            if shuffled_img is not None:
                neural_predict = self.neural_predictor(shuffled_img)
            else:
                neural_predict = self.neural_predictor(x) if self.use_head < 0 else self.neural_predictor((self.use_head, x))

            return neural_predict, neural_final, classification_final


class AttackNet(nn.Module):
    def __init__(self, cotrain_net, victim="classification"):
        super(AttackNet, self).__init__()
        self.module = cotrain_net
        self.victim = victim

    def forward(self, x):
        return self.module(x, which_head=self.victim)


class RegLoss(nn.Module):
    def __init__(self, alpha=0.0):
        super(RegLoss, self).__init__()
        self.alpha = alpha
        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.MSELoss()

    def forward(self, neural_predict, neural_out, outputs, targets):
        loss1 = (1 - self.alpha) * self.criterion1(outputs, targets)
        loss2 = self.alpha * self.criterion2(neural_predict, neural_out)
        return loss1,  loss2

# Both X, Y has size #images times #voxels,
def correlation(X, Y):
    inner = torch.einsum("ab,ab->a", X, Y)
    norm_X = torch.sqrt(torch.einsum("ab,ab->a", X, X))
    norm_Y = torch.sqrt(torch.einsum("ab,ab->a", Y, Y))
    corr = inner / (norm_X * norm_Y)
    return corr
