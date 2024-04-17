"""
convert nifti roi masks into binary pickle files
"""
import argparse
import numpy as np
import os
import utils

parser = argparse.ArgumentParser(description='turn roi masks into pkl')
parser.add_argument('--sub', type=str,
                    help='subject ID: [sub1], [sub2], ...')
parser.add_argument('--roi', type=str,
                    help='roi name: [V1], [V2], [V4],...')
parser.add_argument('--data-dir', default='', type=str,
                    help='directory with ROI masks')


def main(args):

    roi_mask = utils.load_from_nii(os.path.join(args.data_dir, f"{args.sub}_{args.roi}.nii")).flatten()
    roi_arr = np.zeros(shape=roi_mask.shape, dtype=bool)
    cur_roi = np.logical_or(roi_arr, np.isclose(roi_mask, 1))

    utils.pickle_dump(cur_roi, os.path.join(args.data_dir, f"{args.sub}_{args.roi}.pkl"))


if __name__ == "__main__":
    args = parser.parse_args()
    args.data_dir = f"{args.sub}_nsd" if args.data_dir == '' else args.data_dir

    utils.show_input_args(args)

    main(args)