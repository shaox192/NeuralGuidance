
import numpy as np
import argparse

import torch

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
parser.add_argument('--base_model_pth', type=str, help='path to a base neural predictor')
parser.add_argument('--target_model_pth', type=str, help='path to a target neural predictor')

parser.add_argument('--neural_predictor_pos', default="layer4", type=str,
                    help='[layer1], [layer2], [layer3], [[layer4]]')
parser.add_argument('--neural_predictor_arch', default="resnet18", type=str,
                    help='alexnet, [[resnet18]]')
parser.add_argument('--arch', default='resnet18', type=str, help="classifier arch")
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')


def foolbox_attack(val_loader, base_model, target_model, epsilon):
    f_base_model = PyTorchModel(base_model, bounds=(-3.0, 3.0))
    f_target_model = PyTorchModel(target_model, bounds=(-3.0, 3.0))
    attack = LinfPGD()

    base_original_acc_sum = 0.
    target_original_acc_sum = 0.
    n = 0
    epsilons = [epsilon]
    base_perturb_accs = [0. for _ in epsilons]
    target_perturb_accs = [0. for _ in epsilons]
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        base_clean_acc = accuracy(f_base_model, images, target)
        target_clean_acc = accuracy(f_target_model, images, target)
        n += len(images)
        base_original_acc_sum += base_clean_acc * len(images)
        target_original_acc_sum += target_clean_acc * len(images)

        print(f"[{i + 1}]/[{len(val_loader)}]")
        print(f"For base: clean accuracy:  {base_clean_acc * 100:.2f}, avg so far: {base_original_acc_sum / n * 100:.2f}")
        print(f"For target: clean accuracy:  {target_clean_acc * 100:.2f}, avg so far: {target_original_acc_sum / n * 100:.2f}")

        raw_advs, clipped_advs, success = attack(f_base_model, images, target, epsilons=epsilons)
        base_robust_accuracy = 1 - success.float().mean(axis=-1)

        target_robust_accuracy = [accuracy(f_target_model, clipped_advs_item, target) for clipped_advs_item in clipped_advs]
        # print("robust accuracy for perturbations with", end=': ')
        for i, (eps, acc) in enumerate(zip(epsilons, base_robust_accuracy)):
            base_perturb_accs[i] += acc.item() * len(images)
            # print(f"Linf norm ≤ {eps:.4f}: {acc.item() * 100:.2f}, avg so far: {base_perturb_accs[i] / n * 100:.2f}")
        for i, (eps, acc) in enumerate(zip(epsilons, target_robust_accuracy)):
            target_perturb_accs[i] += acc.item() * len(images)
            # print(f"Linf norm ≤ {eps:.4f}: {acc.item() * 100:.2f}, avg so far: {target_perturb_accs[i] / n * 100:.2f}")
    return base_original_acc_sum, epsilons, base_perturb_accs, target_perturb_accs, n


def load_val_data(img_folder_pth, img_folder_txt, workers: int):
    img_folder_ls = data_loader.load_img_folder_ls(img_folder_txt)
    _, val_loader, *_ = data_loader.load_data_folder(img_folder_pth, img_folder_ls, False, 256, workers, workers)
    return val_loader


def main():
    args = parser.parse_args()
    EPSILON_LS = np.arange(0.001, 0.02, 0.002)  # [0.001, 0.002]
    print("Starting L-inf attack, Epsilon to be tested: \n{EPISILON_LS}", flush=True)

    ROI = args.roi
    is_shuffle = False

    val_ldr = load_val_data(args.data, args.img_folder_txt, args.workers)
    print(f"Data loaded from {args.data}; selected categories: {args.img_folder_txt}; num batch {len(val_ldr)}", flush=True)

    base_model = utils.instantiate_ROI_model(args.base_model_pth,
                                        args.neural_predictor_pos, args.neural_predictor_arch, args.arch,
                                        device)
    base_model = base_model.to(device).eval()
    print(f"-> {ROI}-Reg model loaded from {args.base_model_pth}.", flush=True)

    target_model = utils.instantiate_ROI_model(args.target_model_pth,
                                        args.neural_predictor_pos, args.neural_predictor_arch, args.arch,
                                        device)
    target_model = target_model.to(device).eval()
    print(f"-> {ROI}-Reg model loaded from {args.target_model_pth}.", flush=True)

    results = []
    for epsilon in EPSILON_LS:
        print(f"\n-> Current Epsilon: {epsilon}")

        base_original_acc_sum, target_original_acc_sum, epsilons, base_perturb_accs, target_perturb_accs, n = foolbox_attack(val_ldr, base_model, target_model, epsilon)
        results.append([
            base_original_acc_sum / n * 100,
            target_original_acc_sum / n * 100,
            base_perturb_accs[0] / n * 100,
            target_perturb_accs[0] / n * 100,
        ])

    for i in range(len(results)):
        print(
            f"ROI {ROI}: base clean accuracy:  {results[i][0]:.2f}, "
            f"ROI {ROI}: target clean accuracy:  {results[i][1]:.2f}, "
            f"Linf norm ≤ {EPSILON_LS[i]:.4f}, "
            f"base perturbed accuracy: {results[i][2]:.2f}",
            f"target perturbed accuracy: {results[i][3]:.2f}"
        )


if __name__ == "__main__":
    main()