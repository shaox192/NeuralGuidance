"""
Follow-up to style-transfer.py, run all models on the transfered datasets.
"""

import argparse
from torchvision import transforms
import torch
from torch.utils.data import Dataset

import utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Using device: {DEVICE} <--")

parser = argparse.ArgumentParser(description='Style transfer dataset test')

parser.add_argument('--data_pkl', type=str, help='path to dataset')

parser.add_argument('--roi', help='<Required> ROI', required=True)

parser.add_argument('--model_pth', type=str, help='path to a neural predictor')
parser.add_argument('--neural_predictor_pos', default="layer4", type=str,
                    help='[layer1], [layer2], [layer3], [[layer4]]')
parser.add_argument('--neural_predictor_arch', default="resnet18", type=str,
                    help='alexnet, [[resnet18]]')
parser.add_argument('--arch', default='resnet18', type=str, help="classifier arch")


class StyleTransferredDataset(Dataset):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    T = transforms.Compose([transforms.ToTensor(), 
                            transforms.Normalize(mean=MEAN, std=STD)])

    def __init__(self, images, content_lb, style_lb):
        self.images = images
        self.content_lb = content_lb
        
        self.style_lb = style_lb

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = self.images[idx]
        im = self.T(im)

        s_lb, c_lb = self.style_lb[idx], self.content_lb[idx]

        return im, s_lb, c_lb


def test(val_loader, model):

    style_acc1_sum, style_acc5_sum = 0.0, 0.0
    content_acc1_sum, content_acc5_sum = 0.0, 0.0

    n = 0

    for i, (images, style_lb, content_lb) in enumerate(val_loader):
        images, style_lb, content_lb = images.to(DEVICE), style_lb.to(DEVICE), content_lb.to(DEVICE)

        output = model(images)

        s_acc1, s_acc5 = utils.accuracy(output, style_lb, topk=(1, 5))
        c_acc1, c_acc5 = utils.accuracy(output, content_lb, topk=(1, 5))
        
        n += len(images)
        style_acc1_sum += s_acc1 * len(images)
        style_acc5_sum += s_acc5 * len(images)
        content_acc1_sum += c_acc1 * len(images)
        content_acc5_sum += c_acc5 * len(images)

        print(f"Batch [{i}]/[{images.size()[0]}]: \n"
              f"\tcontent acc1: {c_acc1.item()}; content acc5:{c_acc5.item()}\n"
              f"\tstyle acc1: {s_acc1.item()}; style acc5: {s_acc5.item()}")


    return style_acc1_sum / n , style_acc5_sum / n, content_acc1_sum / n, content_acc5_sum / n


def main():
    args = parser.parse_args()
    # =========== validation image
    valdata = utils.pickle_load(args.data_pkl)
    print(f"==> Loaded data from: {args.data_pkl}")
    images = valdata["img"]
    style_labels = valdata["style_lb"]
    content_labels = valdata["content_lb"]

    dataset = StyleTransferredDataset(images, style_labels, content_labels)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    # ============ model
    print(f"==> Loaded model for roi {args.roi}, from file: {args.model_pth}")
    model = utils.instantiate_ROI_model(args.model_pth,
                                        args.neural_predictor_pos, args.neural_predictor_arch, args.arch,
                                        DEVICE)

    model = model.to(DEVICE).eval()

    s_acc1, s_acc5, c_acc1, c_acc5 = test(val_loader, model)
    print(f"Final accuracy fo ROI {args.roi}: \n"
          f"\tcontent acc1: {c_acc1.item()}; content acc5: {c_acc5.item()}\n"
          f"\tstyle acc1: {s_acc1.item()}; style acc5: {s_acc5.item()}")

    return s_acc1.item(), s_acc5.item(), c_acc1.item(), c_acc5.item()


if __name__ == "__main__":
    main()
