"""
Extract manifold statistics (VAF AND RADIUS)
This is splitted from the origginal cotrain_man scripts, can now run separately on delta cpu.
"""


import argparse
import os
import random
import time
import warnings
import numpy as np
import datetime

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

import man_utils
import man_data_loader


parser = argparse.ArgumentParser(description='PyTorch imagenet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenette2',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--img_folder_txt', type=str, help='path to a textfile of image folders used')
parser.add_argument('--seed', type=int, help='seed for random number generator')

parser.add_argument('--batch-size', type=int, help="batch size")

######### original space tuned NP feats
parser.add_argument('--orig-np-feats', type=str, help='path to the pkl with original space neural predictor feats')
parser.add_argument('--var', type=float, default=0.95, help='variance explained threshold')
parser.add_argument('--ndims', type=int, default=None, help='number of dimensions to keep, if not None, var is ignored')

########## subj/roi parameter
parser.add_argument('--roi', default='V1', type=str,
                    help='roi name: [V1], [hV4]...')
parser.add_argument('--sub', type=str, help='subject id')

########## other parameters
parser.add_argument('--data_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')



def build_data_loader(args):
    img_folder_ls = man_data_loader.load_img_folder_ls(args.img_folder_txt)

    ### data Loader
    train_loader, val_loader, train_sampler, val_sampler = \
        man_data_loader.load_data_folder(
            args.data, 
            img_folder_ls, 
            False, 
            args.batch_size,
            args.data_workers, 
            1
        )

    print(f"Data loaded: train: {len(train_loader)}, val: {len(val_loader)}", flush=True)
    
    return train_loader, val_loader, train_sampler, val_sampler


#################### training with orig space VAF + radius ###################
def get_man_stats_from_np_feats(orig_np_feats_f, class_to_idx, var_exp, ndims):
    dat = man_utils.pickle_load(orig_np_feats_f)
    
    gt_lb = np.asarray([lb[0] for lb in dat["gt_lb"]])
    feats = dat["np_out"]

    man_stats = {"center": [None for _ in range(len(class_to_idx))],
                 "basis": [None for _ in range(len(class_to_idx))],
                 "rad" : [None for _ in range(len(class_to_idx))],}
    
    for i, (k, v) in enumerate(class_to_idx.items()):
        man_utils.print_safe(f"[{i+1}]/[{len(class_to_idx)}]-Class {k} has: ")
        idx = np.where(gt_lb == k)
        if len(idx) == 0:
            raise ValueError(f"Class {k} not found in the data!")
        curr_feats = feats[idx]
        man_utils.print_safe(f"\tshape: {curr_feats.shape}")

        ## center
        curr_center = np.mean(curr_feats, axis=0)  # (1084,)

        ## radius
        #### centering performed inside calc_radius
        curr_rad = man_utils.calc_radius(curr_feats.T, 1.)
        man_utils.print_safe(f"\tRMS radius: {curr_rad:.5f}")

        ## basis
        tic = time.time()
        curr_feats -= np.expand_dims(curr_center, axis=0)
        U, V, var_explained = man_utils.calc_svd_dim(curr_feats.T)

        num_d = ndims if ndims else np.sum(var_explained < var_exp) + 1
        man_utils.print_safe(f"\tdim: {num_d} out of {var_explained.shape[0]} dimensions, "
                             f"variance explained: {var_explained[num_d - 1]:.5f}, time: {time.time() - tic:.2f}s")

        U_basis = np.matmul(U[:, :num_d], U[:, :num_d].T)

        assert man_stats["center"][v] is None, f"Manifold stats for class {k} already filled???"
        man_stats["center"][v] = curr_center
        man_stats["rad"][v] = curr_rad
        man_stats["basis"][v] = U_basis
   
    man_stats["center"] = torch.tensor(np.vstack(man_stats["center"]), dtype=torch.float32)
    man_stats["basis"] = torch.tensor(np.stack(man_stats["basis"], axis=0), dtype=torch.float32)
    man_stats["rad"] = torch.tensor(np.asarray(man_stats["rad"]), dtype=torch.float32)

    for k, v in man_stats.items():
        print(k, v.shape)

    return man_stats


def main():
    args = parser.parse_args()

    man_utils.print_safe("\n***check params ---------")
    for arg in vars(args):
        man_utils.print_safe(f"{arg}: {getattr(args, arg)}")
    man_utils.print_safe("--------------------------\n")

    output_dir = f"./{args.sub}_manifold"
    if not os.path.exists(output_dir):
        print(f"Making output dir at {output_dir}")
        os.makedirs(output_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # if you use multiple GPUs
        warnings.warn(f'SEEDING: seed {args.seed}')

    ## -- load data
    train_loader, _, _, _ = build_data_loader(args)

    ## -- load manifold stats
    man_stats = get_man_stats_from_np_feats(args.orig_np_feats, train_loader.dataset.class_to_idx, var_exp=args.var, ndims=args.ndims)

    f_save = os.path.join(output_dir, f"{args.roi}_manifold_stats")
    if args.ndims:
        f_save += f"_ndims{args.ndims}.pkl"
    else:
        f_save += f"_var{args.var:2f}.pkl"
    man_utils.pickle_dump(man_stats, f_save)
    man_utils.print_safe(f"Manifold stats saved to {f_save}, exiting...")


if __name__ == '__main__':
    main()