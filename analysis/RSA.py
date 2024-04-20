import argparse
import time
import torch

import data_loader
import utils

import numpy as np
import h5py

from torch.utils.data import DataLoader
from CustomDataloader import CustomDataloader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Using device: {device} <--")

parser = argparse.ArgumentParser(description='Extract data for RDMs')

parser.add_argument('--roi', help='<Required> ROI', required=True)
parser.add_argument('--wd', type=str, default="no_wd")

parser.add_argument('--save-dir', type=str, help='path to save the checkpoints')
parser.add_argument('--beta-pth', type=str, help='path to neural beta file')
parser.add_argument('--img-pth', type=str, help='path to img h5py')
parser.add_argument('--store-neural', action='store_true', help="dump beta too")


parser.add_argument('--model_pth', type=str, help='path to a neural predictor')
parser.add_argument('--neural_predictor_pos', default="layer4", type=str,
                    help='[layer1], [layer2], [layer3], [[layer4]]')
parser.add_argument('--neural_predictor_arch', default="resnet18", type=str,
                    help='alexnet, [[resnet18]]')
parser.add_argument('--arch', default='resnet18', type=str, help="classifier arch")
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')


activations = []
def get_activations():
    def hook(model, input, output):
        activations.append(output.detach().squeeze())
    return hook

def run_clean(val_loader, model):

    full_lb, full_neural_outs = [], []

    n = 0
    tic = time.time()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            neural_outs = model(images)

            full_lb.append(labels)
            full_neural_outs.append(neural_outs)

            n += len(images)

            if i % 3 == 0:
                print(f"{i}/{len(val_loader)}")
                print(f"* Time elapsed: {time.time() - tic}")
                tic = time.time()

        full_lb = torch.cat(full_lb, dim=0)
        full_neural_outs = torch.cat(full_neural_outs, dim=0)

        return full_lb, full_neural_outs


def get_dist_matrix(data_mat, method):

    if method == "corr":
        # Normalize each row to have zero mean
        mean_centered_data = data_mat - data_mat.mean(dim=1, keepdim=True)

        # Compute the norms of the mean-centered rows
        norms = mean_centered_data.norm(dim=1, keepdim=True)

        # Compute the dot product between each pair of rows
        dot_product = torch.mm(mean_centered_data, mean_centered_data.t())

        # Normalize the result to get the correlation coefficients
        mat = 1. - (dot_product / torch.mm(norms, norms.t()))

    elif method == "euclidean":
        squared_diffs = (data_mat[:, None] - data_mat[None, :]) ** 2
        squared_l2_distances = squared_diffs.sum(dim=-1)

        # Take the square root to get the L2 distances
        mat = torch.sqrt(squared_l2_distances)
    else:
        raise

    return mat


def get_model_corr_mat(lbs):

    # Create a boolean matrix where each entry (i, j) is True if row i and row j have the same label
    same_label_matrix = lbs[:, None] == lbs[None, :]

    # Convert the boolean matrix to the desired float matrix
    model_matrix = torch.where(same_label_matrix, 0.01, 0.99)

    return model_matrix



def load_val_data(img_folder_pth, img_folder_txt, workers: int):
    img_folder_ls = data_loader.load_img_folder_ls(img_folder_txt)
    _, val_loader, *_ = data_loader.load_data_folder(img_folder_pth, img_folder_ls, False, 256, workers, workers)
    return val_loader


def data_loader(img, voxel, batch_size, shuffle):
    data = CustomDataloader(img, voxel)
    data_ldr = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_ldr

def main():
    args = parser.parse_args()
    utils.show_input_args(args)

    ROI = args.roi

    # neural data:
    if ROI in ["None", "random", "V1_shuffle", "TO_shuffle"]:
        args.beta_pth = "V1".join(args.beta_pth.split(ROI))
        args.store_neural = False
        print(f"ROI {ROI}, swaping neural file to V1: {args.beta_pth}", flush=True)

    order_data_orig = utils.pickle_load(args.beta_pth)

    val_img_id = order_data_orig["val_imgID"]
    val_beta = order_data_orig["val_beta"]

    image_data_h5py = h5py.File(args.img_pth, 'r')
    image_data = np.copy(image_data_h5py['stimuli'])
    image_data_h5py.close()
    print(f"-> Loaded stim images from {args.img_pth}, image data shape: {image_data.shape}")

    # extract validation set
    val_images = image_data[val_img_id]
    print(f"*** Validation set, image shape: {val_images.shape}", flush=True)

    val_data_ldr = data_loader(val_images, val_beta, 128, False)
    model = utils.instantiate_ROI_model(args.model_pth,
                                        args.neural_predictor_pos, args.neural_predictor_arch, args.arch,
                                        device)

    model = model.to(device).eval()
    print(f"-> model loaded from {args.model_pth}.", flush=True)

    model.module.shared_layer[8].register_forward_hook(get_activations())

    full_lb, full_neural_outs = run_clean(val_data_ldr, model)
    pool_activations = torch.cat(activations, dim=0)

    print(f"neural head outs: {full_neural_outs.size()}, avgpool outs: {pool_activations.size()}", flush=True)

    if args.store_neural:
        utils.pickle_dump({"val_beta": val_beta}, f"{args.save_dir}/{ROI}_valbeta.pkl")

    f_save = f"{args.save_dir}/{ROI}_outs4RSA.pkl" if args.wd == "no_wd" else f"{args.save_dir}/wd{args.wd}_outs4RSA.pkl"
    utils.pickle_dump({"neural_head": full_neural_outs, "pool_outs": pool_activations}, f_save)




    # sorted_labels, indices = torch.sort(full_lb)
    # sorted_data = full_neural_outs[indices]
    #
    # data_mat = get_dist_matrix(sorted_data, args.dist_method).cpu().numpy()
    # model_mat = get_model_corr_mat(sorted_labels).cpu().numpy()
    #
    # utils.pickle_dump({"data_mat": data_mat, "model_mat": model_mat},
    #                   f"{ROI}_{args.victim}_RSA_mats.pkl")
    #
    # rho, p_value = spearmanr(data_mat.flatten(), model_mat.flatten())
    #
    # print(f"Spearman's rho: {rho}")
    # print(f"P-value: {p_value}")


if __name__ == "__main__":
    main()






