"""
Take all betas, mask the activity using ROIs (to reduce size) and spit out 1 file for all activities
"""
import os
import argparse
from scipy.io import loadmat
from sklearn.decomposition import PCA

import numpy as np
from utils import pickle_load, pickle_dump, load_from_nii, make_directory

parser = argparse.ArgumentParser(description='extract beta from a given roi mask')
parser.add_argument('--sub', default='sub1', type=str,
                    help='sub1, sub2, ...')
parser.add_argument('--roi', default='prf_hV4', type=str,
                    help='roi name: [prf_V1], [prf_hV4]...')
parser.add_argument('--avg_repeat', dest='avg_repeat', action='store_true',
                    help='Average repeated images')


def load_beta_file(filename, voxel_mask):
    values = load_from_nii(filename).transpose((3, 0, 1, 2))
    print(f"\n--> loading from file: {filename}...\n",
          "\tData description: ", values.dtype, np.min(values), np.max(values), values.shape)

    beta = values.reshape((len(values), -1)).astype(np.float32) / 300.
    beta = beta[:, voxel_mask.flatten()]
    print(f"\tBeta size after masked: {beta.shape}")

    return beta


def summarize_repetition(unique_image_dict):
    rep_times = {}
    for k, v in unique_image_dict.items():
        num_rep = len(v)
        if num_rep in rep_times:
            rep_times[num_rep] += 1
        else:
            rep_times[num_rep] = 1
    print(f"--> Repetition summary: ")
    for k, v in rep_times.items():
        print(f"\tnumber of images that repeated for {k} times: {v}")


def avg_repeated_image(full_beta_data, ordering):

    # ---- find images that repeated
    unique_img_dict = {}
    for trial_id in range(len(ordering)):
        img_id = ordering[trial_id]
        if img_id not in unique_img_dict:
            unique_img_dict[img_id] = [trial_id]
        else:
            unique_img_dict[img_id].append(trial_id)
    summarize_repetition(unique_img_dict)

    # ----- average
    new_ordering = []
    new_beta = []
    for img_id, trial_id_ls in unique_img_dict.items():
        if len(trial_id_ls) == 1:
            activation = full_beta_data[trial_id_ls[0]]
        else:
            activations = [full_beta_data[i] for i in trial_id_ls]
            activation = np.mean(activations, axis=0)
        new_ordering.append(img_id)
        new_beta.append(activation)

    new_beta = np.vstack(new_beta)
    new_ordering = np.asarray(new_ordering)
    print(f"\n*** After averaging, {ordering.shape[0]} images reduced to {new_ordering.shape[0]} images,"
          f"beta shape: {new_beta.shape} ***\n")

    return new_ordering, new_beta


def main(sub, roi, calc_repeat):
    prefix = "."
    save_dir = f"{prefix}/{sub}_betas_masked"
    make_directory(save_dir)

    mask_f = f"{prefix}/roi/{sub}_{roi}_roi.pkl"
    mask = pickle_load(mask_f)
    print(f"---> For subject: {sub}: loaded masks from {mask_f}..."
          f"\t actual roi size: {np.sum(mask)}, np array shape: {mask.shape}")

    full_beta_sessions = []
    beta_dir = f"{prefix}/{sub}_betas"
    for beta_f in sorted(os.listdir(beta_dir)):
        if not beta_f.endswith(".nii.gz"):
            continue
        beta = load_beta_file(f"{prefix}/{sub}_betas/{beta_f}", mask)
        full_beta_sessions.append(beta)

    full_beta_sessions = np.concatenate(full_beta_sessions, axis=0)
    print(f"\n*** Final full beta size (num_trial X roi size): {full_beta_sessions.shape} ***\n")

    # ordering
    exp_design = loadmat(f"{prefix}/nsd_expdesign.mat")
    ordering = exp_design['masterordering'].flatten() - 1
    ordering = ordering[:full_beta_sessions.shape[0]]  # in case we are only using a subset of sessions

    if calc_repeat:
        print("---> Averaging repeated images...")
        ordering, full_beta_sessions = avg_repeated_image(full_beta_sessions, ordering)
        dump_f = f"{save_dir}/{sub}_orders_betas_{roi}.pkl"
    else:
        dump_f = f"{save_dir}/{sub}_betas_{roi}.pkl"

    val_mask = ordering < 1000

    val_id = ordering[val_mask]
    val_data = full_beta_sessions[val_mask]

    train_id = ordering[~val_mask]
    train_data = full_beta_sessions[~val_mask]

    mb = np.mean(train_data, axis=0, keepdims=True)
    sb = np.std(train_data, axis=0, keepdims=True)

    train_data = np.nan_to_num((train_data - mb) / (sb + 1e-6))
    val_data = np.nan_to_num((val_data - mb) / (sb + 1e-6))

    # perform PCA on the data
    print(f"\n>>*** Beta before PCA: train-{train_data.shape}, val-{val_data.shape} ***<<\n")
    pca = PCA(n_components=0.95, svd_solver="full")
    pca.fit(train_data)
    train_data = pca.transform(train_data)
    val_data = pca.transform(val_data)
    print(f"\n>>*** Beta after PCA: train-{train_data.shape}, val-{val_data.shape}; "
          f"Variance Explained: {pca.explained_variance_ratio_.cumsum()[-1]}% ***<<\n")

    betaorder2save = {"val_imgID": val_id, "val_beta": val_data,
                      "train_imgID": train_id, "train_beta": train_data}
    pickle_dump(betaorder2save, dump_f)


if __name__ == "__main__":
    args = parser.parse_args()
    sub = args.sub
    roi = args.roi
    calc_repeat = args.avg_repeat

    main(sub, roi, calc_repeat)
