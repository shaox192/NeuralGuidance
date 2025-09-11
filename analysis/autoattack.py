
import argparse
import torch

import data_loader
import utils
from autoattack import AutoAttack

import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Using device: {device} <--")


parser = argparse.ArgumentParser(description='Autoattack')

parser.add_argument('--roi', help='ROI', required=True)
parser.add_argument('--img_folder_txt', type=str, help='path to a textfile of image folders used')
parser.add_argument('--data', type=str, help='path to dataset')
parser.add_argument('--model_pth', type=str, help='path to a neural predictor')

parser.add_argument('--neural_predictor_pos', default="layer4", type=str,
                    help='[layer1], [layer2], [layer3], [[layer4]]')
parser.add_argument('--neural_predictor_arch', default="resnet18", type=str,
                    help='alexnet, [[resnet18]]')
parser.add_argument('--arch', default='resnet18', type=str, help="classifier arch")
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')



def load_reg_data(img_folder_pth, img_folder_txt, workers: int):
    img_folder_ls = data_loader.load_img_folder_ls(img_folder_txt)
    _, val_loader, *_ = data_loader.load_data_folder(img_folder_pth, img_folder_ls, False, 256, workers, workers)
    return val_loader


def main():

    args = parser.parse_args()

    print("\n***check params ---------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("--------------------------\n")

    EPSILON_LS = np.arange(0.001, 0.02, 0.002)
    print(f"Starting AUTO-attack, Epsilon to be tested: \n{EPSILON_LS}", flush=True)

    val_ldr = load_reg_data(args.data, args.img_folder_txt, args.workers)
    print(f"Data loaded from: {args.data}: num batch {len(val_ldr)}", flush=True)

    l = [x for (x, y) in val_ldr]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in val_ldr]
    y_test = torch.cat(l, 0)


    model = utils.instantiate_ROI_model(args.model_pth, args.neural_predictor_pos, args.neural_predictor_arch, args.arch, device)
    print(f"-> model loaded from {args.model_pth}.", flush=True)

    model = model.to(device).eval()

    for e_i in range(EPSILON_LS.shape[0]):
        max_eps = EPSILON_LS[e_i]
        print(f"---------- Attack {e_i}: {max_eps}", flush=True)
        adversary = AutoAttack(model, norm="Linf", eps=max_eps,
                               log_path=f"./{args.roi}_epsi{e_i}.txt",
                               version='standard')
        adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=256)


if __name__ == "__main__":
    main()