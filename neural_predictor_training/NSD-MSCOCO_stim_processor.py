"""
Entry point for splitting the images data into all 8 subjects' separately
Will download the dataset if not already present.
"""

import sys
import os
import requests
import subprocess
import argparse

import numpy as np
import h5py
from scipy.io import loadmat
import PIL.Image as pim
import pickle

import utils


def save_splitted_data(save_to_this_file, data_objects_dict):
    failed = []
    with h5py.File(save_to_this_file + '.h5py', 'w') as hf:
        for k, v in data_objects_dict.items():
            try:
                hf.create_dataset(k, data=v)
                print('saved %s in h5py file' % (k))
            except:
                failed.append(k)
                print('failed to save %s as h5py. will try pickle' % (k))
    for k in failed:
        with open(save_to_this_file + '_' + '%s.pkl' % (k), 'w') as pkl:
            try:
                pickle.dump(data_objects_dict[k], pkl)
                print('saved %s as pkl' % (k))
            except:
                print('failed to save %s in any format. lost.' % (k))


def resize_image_tensor(x, newsize):
    tt = x.transpose((0,2,3,1))
    r  = np.ndarray(shape=x.shape[:1]+newsize+(x.shape[1],), dtype=tt.dtype) 
    for i,t in enumerate(tt):
        r[i] = np.asarray(pim.fromarray(t).resize(newsize, resample=pim.BILINEAR))
    return r.transpose((0,3,1,2))   


def main():
    output_dir = f"./nsd_coco_stimuli"
    if not os.path.exists(output_dir):
        print(f"Making output dir at {output_dir}")
        os.makedirs(output_dir)
    orig_coco_stimuli_file = f"{output_dir}/nsd_stimuli.hdf5"
    if not os.path.exists(orig_coco_stimuli_file):
        print("--- Downloading original coco stimuli file... THIS IS LARGE (37GB), TAKES TIME.")
        url = "https://natural-scenes-dataset.s3.amazonaws.com/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
        subprocess.run(["wget", url, "-O", orig_coco_stimuli_file])
        print("--- Download complete.")

    exp_design_file = f"./sub1_nsd/nsd_expdesign.mat"
    exp_design = loadmat(exp_design_file)
    print("Check exp_design mat loaded: ", exp_design.keys())

    shared_idx = exp_design['sharedix']
    subject_idx = exp_design['subjectim']
    trial_order = exp_design['masterordering']

    print("Check exp_design mat trial order: ", np.min(trial_order), np.max(trial_order))

    stim_file = orig_coco_stimuli_file
    print("Loading block...")
    image_data_set = h5py.File(stim_file, 'r')
    print("Check all stimuli file loaded: ", image_data_set.keys())
    image_data = np.copy(image_data_set['imgBrick'])
    image_data_set.close()
    print("Check image data copied: ", image_data.shape)

    img_new_size = 227
    for k,s_idx in enumerate(subject_idx):
        s_image_data = image_data[s_idx - 1]
        s_image_data = resize_image_tensor(s_image_data.transpose(0,3,1,2), newsize=(img_new_size,img_new_size))

        print(s_image_data.shape)
        save_splitted_data(f"{output_dir}/sub{k+1}_stimuli_{img_new_size}", {'stimuli': s_image_data})


if __name__ == "__main__":
    main()


