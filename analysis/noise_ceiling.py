"""
Take all betas, mask the activity using ROIs nd spit out 1 file for noise ceiling of each voxel.
"""
import os
import argparse
from scipy.io import loadmat
import numpy as np

import utils


parser = argparse.ArgumentParser(description='extract noise ceiling for a ROI')
parser.add_argument('--sub', type=str,
                    help='subject ID: [sub1], [sub2], ...')
parser.add_argument('--roi', type=str,
                    help='roi name: [V1], [V2], [V4],...')
parser.add_argument('--beta-data', type=str)
parser.add_argument('--aux-data', type=str)


def load_beta_file(filename, voxel_mask):
    values = utils.load_from_nii(filename).transpose((3, 0, 1, 2))
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


def get_repeated_image(full_beta_data, ordering):

    # ---- find images that repeated
    unique_img_dict = {}
    for trial_id in range(len(ordering)):
        img_id = ordering[trial_id]
        if img_id not in unique_img_dict:
            unique_img_dict[img_id] = [trial_id]
        else:
            unique_img_dict[img_id].append(trial_id)

    use_dict = {k: v for k, v in unique_img_dict.items() if len(v) == 3}  # WE ONLY USE IMAGES WITH 3 reps for this!!
    return use_dict

def cal_noise_ceiling_single_vox(data):
    ## data: [each image * 3 reps]

    # Step 1: Calculate the variance of the betas for each image
    variances = np.var(data, axis=1, ddof=1)

    # Step 2: Compute the noise standard deviation
    # Average the variance across images and take the square root
    noise_std = np.sqrt(np.mean(variances))

    # Since we have z-scored data, the total variance is 1,
    # thus the signal standard deviation can be calculated as follows:
    signal_std = np.sqrt(np.maximum(1 - noise_std ** 2, 0))

    # Step 3: Compute the noise ceiling SNR
    ncsnr = signal_std / noise_std

    noise_ceiling = (ncsnr ** 2 / (ncsnr ** 2 + 1 / 3.)) * 100

    return noise_ceiling


def main(sub, roi, beta_data, aux_data):

    mask_f = f"{aux_data}/{sub}_{roi}.pkl"
    mask = utils.pickle_load(mask_f)
    print(f"*** loaded masks from {mask_f}. Actual roi size: {np.sum(mask)}, np array shape: {mask.shape}")

    full_beta_sessions = []
    for beta_f in sorted(os.listdir(beta_data)):
        if not beta_f.endswith(".nii.gz"):
            continue
        beta = load_beta_file(os.path.join(beta_data, beta_f), mask)

        ## standardize within each session for each voxel
        means = np.mean(beta, axis=0, keepdims=True)
        stds = np.std(beta, axis=0, keepdims=True)
        beta = (beta - means) / stds

        full_beta_sessions.append(beta)

    full_beta_sessions = np.concatenate(full_beta_sessions, axis=0)
    print(f"\n*** Final full beta size (num_trial X roi size): {full_beta_sessions.shape}")

    # ordering
    exp_design = loadmat(f"{aux_data}/nsd_expdesign.mat")
    ordering = exp_design['masterordering'].flatten() - 1
    ordering = ordering[:full_beta_sessions.shape[0]]  # in case we are only using a subset of sessions

    img_rep_d = get_repeated_image(full_beta_sessions, ordering)
    summarize_repetition(img_rep_d)

    beta_by_voxel = []  # [[[img1_rep1], [img1_rep2], ...], [img2], ...]
    for img_id, trial_id_ls in img_rep_d.items():
        activations = [full_beta_sessions[i] for i in trial_id_ls]
        beta_by_voxel.append(activations)

    full_NC_v = []
    for v in range(full_beta_sessions.shape[1]):
        curr_beta_v = []
        for img_i in range(len(beta_by_voxel)):
            curr_beta_v.append([beta_by_voxel[img_i][rep][v] for rep in range(3)])
        curr_beta_v = np.asarray(curr_beta_v)

        NC_v = cal_noise_ceiling_single_vox(curr_beta_v)
        full_NC_v.append(NC_v)

    final_NC = np.mean(full_NC_v)
    NC_std = np.std(full_NC_v)

    utils.pickle_dump(full_NC_v, f"./{sub}_{roi}_NC.pkl")
    print(f"Final noise ceiling: {final_NC} +- {NC_std}", flush=True)


if __name__ == "__main__":
    args = parser.parse_args()
    utils.show_input_args(args)

    sub = args.sub
    roi = args.roi
    beta_data = args.beta_data
    aux_data = args.aux_data

    main(sub, roi, beta_data, aux_data)
