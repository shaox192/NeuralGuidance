"""
Run style transfer on all validation set
https://github.com/leongatys/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
save the transfered set and corresponding labels
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torchvision import transforms, models
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
from utils import pickle_dump
import data_loader

parser = argparse.ArgumentParser(description='Style transfer dataset prepare')
parser.add_argument('--img_folder_txt', type=str, help='path to a textfile of image folders used')
parser.add_argument('--data', type=str, help='path to dataset')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--save_f', type=str, help='output data save, must end with .pkl')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n*====== USING DEVICE: {DEVICE} ======*\n")


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    # features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
    # G = torch.mm(features, features.t())  # compute the gram product
    G = torch.einsum("abcd,aecd->abe", input, input)

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        # self.loss = torch.sum(self.loss, dim=[1,2])

        return input


# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = cnn_normalization_mean.view(-1, 1, 1)
        self.std = cnn_normalization_std.view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std


def get_style_model_and_losses(style_img, content_img):
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    cnn = models.vgg19(pretrained=True).features.eval()
    cnn.to(DEVICE)
    # normalization module
    normalization = Normalization()

    # losses
    content_losses = []
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(model, style_losses, content_losses, input_img, num_steps):
    """Run the style transfer."""
    # weights:
    style_weights = [1e10 / n ** 2 for n in [64, 128, 256, 512, 512]]

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img])

    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0.0  # torch.zeros(size=input_img.size()[0])
            content_score = 0.0  # torch.zeros(size=input_img.size()[0])

            for i in range(len(style_losses)):
                style_score += style_losses[i].loss * style_weights[i]
            for cl in content_losses:
                content_score += cl.loss

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"\trun {run} - Style Loss : {style_score.item()} Content Loss: {content_score.item()}")

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def load_img_pair(img_pair_ls):
    PREP_T = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])
    im_ls = []
    for im_p in img_pair_ls:
        im = Image.open(im_p)
        im = PREP_T(im).unsqueeze(0)
        im.to(DEVICE, torch.float)
        im_ls.append(im)

    style_image, content_image = im_ls
    input_img = content_image.clone()
    return style_image, content_image, input_img


def imsave(im_T, save_f=None):
    image = im_T.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = transforms.ToPILImage()(image)
    if save_f is not None:
        image.save(save_f)
    return image


class StyleContentDataset(Dataset):
    PREP_T = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])

    def __init__(self, val_img_f_ls, style_idx_ls, content_idx_ls):
        self.val_img_f_ls = val_img_f_ls
        self.style_idx_ls = style_idx_ls
        self.content_idx_ls = content_idx_ls

        self.style_img_f_ls = [val_img_f_ls[i] for i in style_idx_ls]
        self.content_img_f_ls = [val_img_f_ls[i] for i in content_idx_ls]

    def __len__(self):
        return len(self.val_img_f_ls)

    def __getitem__(self, idx):
        style_img, style_lb = self.style_img_f_ls[idx]
        style_img = self.PREP_T(Image.open(style_img).convert("RGB"))

        content_img, content_lb = self.content_img_f_ls[idx]
        content_img = self.PREP_T(Image.open(content_img).convert("RGB"))

        return style_img, style_lb, content_img, content_lb


def load_val_data(img_folder_pth, img_folder_txt, workers: int):
    img_folder_ls = data_loader.load_img_folder_ls(img_folder_txt)
    _, val_loader, *_ = data_loader.load_data_folder(img_folder_pth, img_folder_ls, False, 256, workers, workers)
    return val_loader


def style_content_dataloader(data, img_folder_txt, workers, batch_size):
    np.random.seed(1024)
    val_dataset = load_val_data(data, img_folder_txt, workers).dataset
    # val_dataset = datasets.ImageFolder(val_img_dir)

    content_full_lb = val_dataset.targets
    style_full_idx = []
    for lb_i in range(len(content_full_lb)):
        style_i = lb_i
        while content_full_lb[style_i] == content_full_lb[lb_i]:
            style_i = np.random.choice(np.arange(len(content_full_lb)), replace=False)
        style_full_idx.append(style_i)

    dataset = StyleContentDataset(val_dataset.samples, style_full_idx, np.arange(len(content_full_lb)))

    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return val_loader


def main():
    args = parser.parse_args()

    # img_dir = "/scratch/bbtb/imagenet50_textshape/val"  # "../texture_shape_img"

    transferred_img_set = {"img": [], "style_lb": [], "content_lb": []}
    val_loader = style_content_dataloader(args.data, args.img_folder_txt, args.workers, batch_size=64)

    for i, (style_img, style_lb, content_img, content_lb) in enumerate(val_loader):
        print(f"\nProcessing Img Batch number {i}...", flush=True)

        input_img = content_img.clone()
        style_img, content_img, input_img = style_img.to(DEVICE), content_img.to(DEVICE), input_img.to(DEVICE)

        model, style_losses, content_losses = get_style_model_and_losses(style_img, content_img)

        output = run_style_transfer(model, style_losses, content_losses, input_img, num_steps=200)

        transferred_img_set["img"] += [imsave(output[i]) for i in range(output.size()[0])]
        transferred_img_set["style_lb"] += style_lb.tolist()
        transferred_img_set["content_lb"] += content_lb.tolist()

    pickle_dump(transferred_img_set, args.save_f)


if __name__ == "__main__":
    main()

