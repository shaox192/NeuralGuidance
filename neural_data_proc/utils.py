import os
import pickle as pkl
import h5py

def make_directory(pth):
    if not os.path.exists(pth):
        print(f"Making output dir at {pth}")
        os.makedirs(pth)
    else:
        print(f"Path {pth} exists.")


def load_from_nii(mask_nii_file):
    import nibabel as nib
    return nib.load(mask_nii_file).get_fdata()


def pickle_load(fpth):
    print(f"loading from: {fpth}")
    with open(fpth, 'rb') as f:
        return pkl.load(f)


def pickle_dump(data, fpth):
    print(f"writing to: {fpth}")
    with open(fpth, 'wb') as f:
        pkl.dump(data, f)


def save_stuff(save_to_this_file, data_objects_dict):
    failed = []
    with h5py.File(save_to_this_file+'.h5py', 'w') as hf:
        for k,v in data_objects_dict.items():
            try:
                hf.create_dataset(k,data=v)
                print ('saved %s in h5py file' %(k))
            except:
                failed.append(k)
                print ('failed to save %s as h5py. will try pickle' %(k))
    for k in failed:
        with open(save_to_this_file+'_'+'%s.pkl' %(k), 'w') as pkl:
            try:
                pkl.dump(data_objects_dict[k],pkl)
                print ('saved %s as pkl' %(k))
            except:
                print ('failed to save %s in any format. lost.' %(k))