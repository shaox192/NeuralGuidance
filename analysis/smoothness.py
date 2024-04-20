import numpy as np
import argparse

import torch
import torch.autograd as autograd

from foolbox import PyTorchModel, accuracy
from foolbox.attacks import LinfPGD

import utils
import data_loader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Using device: {device} <--")

parser = argparse.ArgumentParser(description='Linf attack')
parser.add_argument('--roi', help='<Required> ROI', required=True)
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


def foolbox_linf_attack(model, images, target, epsilons):
    attack = LinfPGD(rel_stepsize=0.01)
    raw_advs, clipped_advs, success = attack(model, images, target, epsilons=epsilons)

    return clipped_advs, success


def Magnitude(g1):
    return (torch.sum(g1 ** 2, 1)).mean() * 2 * 200 # scale up 200 because too small


def attack_main(val_loader, model, epsilon):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    fmodel = PyTorchModel(model, bounds=(-3.0, 3.0))

    original_acc_sum = 0.
    n = 0
    epsilons = [epsilon]

    smooth_loss = [0.0 for _ in epsilons]
    perturb_accs = [0. for _ in epsilons]
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        # clean accuracy
        clean_acc = accuracy(fmodel, images, target)
        n += len(images)
        original_acc_sum += clean_acc * len(images)
        print(f"[{i + 1}]/[{len(val_loader)}]")
        print(f"clean accuracy:  {clean_acc * 100:.2f}, avg so far: {original_acc_sum / n * 100:.2f}")

        # perform the attack
        adv_img, success = foolbox_linf_attack(fmodel, images, target, epsilons)

        # robustness accuracy
        robust_accuracy = 1 - success.float().mean(axis=-1)
        print("robust accuracy for perturbations with", end=': ')
        for i, (eps, acc) in enumerate(zip(epsilons, robust_accuracy)):
            perturb_accs[i] += acc.item() * len(images)
            print(f"Linf norm ≤ {eps:.4f}: {acc.item() * 100:.2f}, avg so far: {perturb_accs[i] / n * 100:.2f}")

        # smoothness measure
        for adv_i in range(len(adv_img)):
            adv_x = adv_img[adv_i]
            adv_x.requires_grad = True

            outputs = model(adv_x)
            loss = criterion(outputs, target)
            grad = autograd.grad(loss, adv_x, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            sm_l = Magnitude(grad).item()
            smooth_loss[adv_i] += sm_l * len(images)
            print(f"smoothness this batch: {sm_l: .4f} avg so far: {smooth_loss[adv_i] / n:.4f}")

    return original_acc_sum, epsilons, perturb_accs, smooth_loss, n


def load_val_data(img_folder_pth, img_folder_txt, workers: int):
    img_folder_ls = data_loader.load_img_folder_ls(img_folder_txt)
    _, val_loader, *_ = data_loader.load_data_folder(img_folder_pth, img_folder_ls, False, 256, workers, workers)
    return val_loader


def main():
    args = parser.parse_args()
    EPSILON_LS = [.08, .12, 0.18, 0.26, 0.38] # np.arange(0.001, 0.02, 0.002)  # [0.001, 0.002]
    print("Starting L-inf attack, Epsilon to be tested: \n{EPISILON_LS}", flush=True)

    ROI = args.roi
    is_shuffle = False

    val_ldr = load_val_data(args.data, args.img_folder_txt, args.workers)
    print(f"Data loaded from {args.data}; selected categories: {args.img_folder_txt}; num batch {len(val_ldr)}",
          flush=True)

    model = utils.instantiate_ROI_model(args.model_pth,
                                        args.neural_predictor_pos, args.neural_predictor_arch, args.arch,
                                        device)
    model = model.to(device).eval()
    print(f"-> {ROI}-Reg model loaded from {args.model_pth}.", flush=True)

    results = []
    for epsilon in EPSILON_LS:
        print(f"\n-> Current Epsilon: {epsilon}")

        original_acc_sum, epsilons, perturb_accs, smoothness, n = attack_main(val_ldr, model, epsilon)
        results.append([original_acc_sum / n * 100, perturb_accs[0] / n * 100, smoothness[0] / n])

    for i in range(len(results)):
        print(
            f"ROI {ROI}: clean accuracy:  {results[i][0]:.2f}, "
            f"Linf norm ≤ {EPSILON_LS[i]:.4f}, "
            f"perturbed accuracy: {results[i][1]:.2f}"
            f"smoothness: {results[i][2]: .2f}"
        )


if __name__ == "__main__":
    main()
