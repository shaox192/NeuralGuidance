"""
generate roi masks as pkl files
nii roi masks were processed by afni --> binary masks for each roi
use extract_roi.sh to do this.
"""
import argparse
import numpy as np
from utils import pickle_dump, load_from_nii

parser = argparse.ArgumentParser(description='turn roi masks into pkl')
parser.add_argument('--sub', default='sub1', type=str,
                    help='sub1, sub2, ...')
parser.add_argument('--roi', default='prf_hV4', type=str,
                    help='roi name: [prf_V1], [prf_hV4]...')



def main(sub, roi):
    roi_dir = "roi"

    roi_mask = load_from_nii(f"{roi_dir}/{sub}_{roi}_roi.nii").flatten()
    roi_arr = np.zeros(shape=roi_mask.shape, dtype=bool)
    cur_roi = np.logical_or(roi_arr, np.isclose(roi_mask, 1))
    pickle_dump(cur_roi, f"{roi_dir}/{sub}_{roi}_roi.pkl")

if __name__ == "__main__":
    args = parser.parse_args()
    roi = args.roi
    sub = args.sub

    main(sub, roi)