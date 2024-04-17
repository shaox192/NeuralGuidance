import os
import pickle as pkl


def make_directory(pth):
    if not os.path.exists(pth):
        print(f"Making output dir at {pth}")
        os.makedirs(pth)
    else:
        print(f"Path {pth} exists.")


def load_from_nii(mask_nii_file):
    import nibabel as nib
    if os.path.exists(mask_nii_file):
        return nib.load(mask_nii_file).get_fdata()
    else:
        raise FileNotFoundError(f"can't find mask file: {mask_nii_file}!")


def pickle_load(fpth):
    print(f"loading from: {fpth}")
    with open(fpth, 'rb') as f:
        return pkl.load(f)


def pickle_dump(data, fpth):
    print(f"writing to: {fpth}")
    with open(fpth, 'wb') as f:
        pkl.dump(data, f)


def show_input_args(args):
    print("\n***check params ---------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("--------------------------\n")
