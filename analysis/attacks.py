
import numpy as np
import argparse

import torch

from foolbox import PyTorchModel, accuracy
from foolbox.attacks import LinfPGD, LinfFastGradientAttack, L2CarliniWagnerAttack, \
    L2DeepFoolAttack, L2FastGradientAttack, L2BrendelBethgeAttack

import utils
import data_loader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Using device: {device} <--")


parser = argparse.ArgumentParser(description='Linf attack')
parser.add_argument('--roi', help='<Required> ROI', required=True)
parser.add_argument('--img_folder_txt', type=str, help='path to a textfile of image folders used')
parser.add_argument('--data', type=str, help='path to dataset')
parser.add_argument('--model-pth', type=str, help='path to a neural predictor')
parser.add_argument('--attack-type', type=int,
                    help='attack type:'
                         '0: LinfPGD, '
                         '1: LinfFastGradientAttack, '
                         '2: L2CarliniWagnerAttack')
parser.add_argument('--batch-size', type=int, default=128, help='validation batch size')


parser.add_argument('--neural_predictor_pos', default="layer4", type=str,
                    help='[layer1], [layer2], [layer3], [[layer4]]')
parser.add_argument('--neural_predictor_arch', default="resnet18", type=str,
                    help='alexnet, [[resnet18]]')
parser.add_argument('--arch', default='resnet18', type=str, help="classifier arch")
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')


ATTACK_TYPE = {0: "LinfPGD", 1: "LinfFastGradientAttack",
               2: "L2CarliniWagnerAttack", 3: "L2DeepFoolAttack",
               4: "L2FastGradientAttack", 5: "L2BrendelBethgeAttack"}

def foolbox_attack(val_loader, model, epsilon, attack_type):
    fmodel = PyTorchModel(model, bounds=(-3.0, 3.0))
    if attack_type == 0:
        attack = LinfPGD()
    elif attack_type == 1:
        attack = LinfFastGradientAttack()
    elif attack_type == 2:
        attack = L2CarliniWagnerAttack()
    elif attack_type == 3:
        attack = L2DeepFoolAttack(steps=40)
    elif attack_type == 4:
        attack = L2FastGradientAttack()
    elif attack_type == 5:
        attack = L2BrendelBethgeAttack()
    else:
        raise "Wrong attack type!"

    original_acc_sum = 0.
    n = 0
    epsilons = [epsilon]
    perturb_accs = [0. for _ in epsilons]
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        clean_acc = accuracy(fmodel, images, target)
        n += len(images)
        original_acc_sum += clean_acc * len(images)
        print(f"[{i + 1}]/[{len(val_loader)}]")
        print(f"clean accuracy:  {clean_acc * 100:.2f}, avg so far: {original_acc_sum / n * 100:.2f}", flush=True)

        try:
            raw_advs, clipped_advs, success = attack(fmodel, images, target, epsilons=epsilons)
        except:
            print("----->>>>>>> Something happened", flush=True)
            exit()
        print("------ finished attack -------", flush=True)

        robust_accuracy = 1 - success.float().mean(axis=-1)
        print("robust accuracy for perturbations with", end=': ')
        for i, (eps, acc) in enumerate(zip(epsilons, robust_accuracy)):
            perturb_accs[i] += acc.item() * len(images)
            print(f"Linf norm ≤ {eps:.4f}: {acc.item() * 100:.2f}, avg so far: {perturb_accs[i] / n * 100:.2f}", flush=True)
    return original_acc_sum, epsilons, perturb_accs, n


def load_val_data(img_folder_pth, img_folder_txt, workers: int, batch_size=128):
    img_folder_ls = data_loader.load_img_folder_ls(img_folder_txt)
    _, val_loader, *_ = data_loader.load_data_folder(img_folder_pth, img_folder_ls, False, batch_size, workers, workers)
    return val_loader


def main():
    args = parser.parse_args()

    print("\n***check params ---------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("--------------------------\n")

    EPSILON_LS = np.arange(0.001, 0.02, 0.002)  # [0.001, 0.002]
    if args.attack_type in [2, 3, 4, 5]:
        EPSILON_LS = np.sqrt(224 * 224 * 3) * EPSILON_LS

    print(f"Starting {ATTACK_TYPE[args.attack_type]} attack, Epsilon to be tested: \n{EPSILON_LS}", flush=True)

    ROI = args.roi

    val_ldr = load_val_data(args.data, args.img_folder_txt, args.workers, args.batch_size)
    print(f"Data loaded from {args.data}; selected categories: {args.img_folder_txt}; num batch {len(val_ldr)}", flush=True)

    model = utils.instantiate_ROI_model(args.model_pth,
                                        args.neural_predictor_pos, args.neural_predictor_arch, args.arch,
                                        device)
    model = model.to(device).eval()
    print(f"-> {ROI}-Reg model loaded from {args.model_pth}.", flush=True)

    results = []
    for epsilon in EPSILON_LS:
        print(f"\n-> Current Epsilon: {epsilon}")

        original_acc_sum, epsilons, perturb_accs, n = foolbox_attack(val_ldr, model, epsilon, args.attack_type)
        results.append([original_acc_sum / n * 100, perturb_accs[0] / n * 100])

    for i in range(len(results)):
        print(
            f"ROI {ROI}: clean accuracy:  {results[i][0]:.2f}, "
            f"Linf norm ≤ {EPSILON_LS[i]:.4f}, "
            f"perturbed accuracy: {results[i][1]:.2f}"
        )


if __name__ == "__main__":
    main()
