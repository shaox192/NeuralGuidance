import os
import torch
import torchvision.transforms as transforms
import numpy as np
from itertools import groupby

from PIL import Image
import pickle as pkl
import time

import man_utils


NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
TRAIN_T = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            NORMALIZE,
        ])
TEST_T = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            NORMALIZE,
        ])


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
    

class ImageFolderSelected:

    def __init__(self, folder_root, folder_ls, transform):
        self.root = folder_root
        self.folder_ls = sorted(folder_ls)

        self.transform = transform
        self.classes, self.class_to_idx, self.idx_to_class = self.find_classes()
        # self.class_to_idx: uses imagnet code: e.g. {n01440764: 0} from the folder!

        self.samples = self.make_dataset(self.class_to_idx)  # this is sorted by class
        self.loader = pil_loader
        self.targets = [s[1] for s in self.samples]

    def _generate_within_shuffled_idx(self, rng):
        """
        generate shuffled index within each class
        """
        idx_out = []
        # Enumerate samples to keep track of indices
        for key, group in groupby(enumerate(self.samples), key=lambda x: x[1][1]):
            indices = [i for i, _ in group]
            rng.shuffle(indices)
            idx_out.extend(indices)
        return idx_out 
            
    def find_classes(self):
        classes = sorted(self.folder_ls)  # !! alphabetical!
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}

        return classes, class_to_idx, idx_to_class
    
    def make_dataset(self, class_to_idx):
        samples = []

        for f in self.folder_ls:
            img_folder_pth = os.path.join(self.root, f)

            for img_f in sorted(os.listdir(img_folder_pth)):
                if img_f.endswith(".JPEG") or img_f.endswith(".jpeg"):
                    img_f_pth = os.path.join(img_folder_pth, img_f)
                    class_i = class_to_idx[f]
                    samples.append((img_f_pth, class_i))
        return samples

    def _load_img(self, pth):
        img = self.loader(pth)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = self._load_img(path)

        return img, target, 0 # dummy for legacy compatibility

    def __len__(self) -> int:
        return len(self.samples)


def load_data_folder(prt_data_pth, folder_names, distributed, batch_size,
                     train_workers, test_workers, 
                     shuffle_mode=None,
                     ):
                     # orig_imagenet_lbs=None):
    """
    load data using ImageFolder for selected folders (categories)
    :param args:
        :shuffle_model: "bl" or "within", bl: baseline, within: within category manifold
    :return:
    """
    traindir = os.path.join(prt_data_pth, 'train')
    valdir = os.path.join(prt_data_pth, 'val')

    train_dataset = ImageFolderSelected(traindir, folder_names, TRAIN_T, shuffle_mode=shuffle_mode) #, orig_imagenet_lbs=orig_imagenet_lbs)
    val_dataset = ImageFolderSelected(valdir, folder_names, TEST_T, shuffle_mode=None)
    
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=train_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=test_workers, pin_memory=True, sampler=val_sampler)
    
    return train_loader, val_loader, train_sampler, val_sampler


def load_img_folder_ls(txt_pth):
    with open(txt_pth, 'r') as f:
        folder_ls = []
        for ln in f.readlines():
            folder_ls.append(ln.split(':')[0].strip())
        return folder_ls


class ImageFolderSelectedManifolds(ImageFolderSelected):
    def __init__(self, folder_root, folder_ls, transform, manstats,
                       shuffle=False):
        super().__init__(folder_root, folder_ls, transform, shuffle=shuffle)
        self.manstats = manstats
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, shuffled_img, w_proj, center, rad, dim) 
            where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = self._load_img(path)
        if self.shuffle:
            idx = self.idx2shuffledidx[index]
            shuffled_path, _ = self.samples[idx]
            shuffled_img = self._load_img(shuffled_path)
        else:
            shuffled_img = 0

        category_name = self.idx_to_class[target]
        man = self.manstats.get_manifold(category_name)

        return img, target, shuffled_img, *man


class ManifoldStats:
    def __init__(self, w_proj, gt_lb, centers, rads, dims):
        self.w_proj = w_proj
        self.gt_lb = gt_lb

        self.centers = centers
        self.rads = rads
        self.dims = dims
    
    def get_manifold(self, category_name):
        cat_idx = np.where(self.gt_lb == category_name)[0]

        return self.w_proj[cat_idx], self.centers[cat_idx], self.rads[cat_idx], self.dims[cat_idx]




#################### training with orig space VAF + radius ###################

def get_man_stats_from_np_feats(orig_np_feats_f, class_to_idx):
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

        num_d = np.sum(var_explained < 0.95) + 1
        man_utils.print_safe(f"\tdim: {num_d} out of {var_explained.shape[0]} dimensions, "
                             f"variance explained: {var_explained[num_d - 1]:.5f}, time: {time.time() - tic:.2f}s")

        U_basis = np.matmul(U[:, :num_d], U[:, :num_d].T)

        assert man_stats["center"][v] is None, f"Manifold stats for class {k} already filled???"
        man_stats["center"][v] = curr_center
        man_stats["rad"][v] = curr_rad
        man_stats["basis"][v] = U_basis
    
    # for k, v in man_stats.items():
    #     print(k, len(v), v[0] if k == "rad" else v[0].shape)
   
    man_stats["center"] = torch.tensor(np.vstack(man_stats["center"]), dtype=torch.float32)
    man_stats["basis"] = torch.tensor(np.stack(man_stats["basis"], axis=0), dtype=torch.float32)
    man_stats["rad"] = torch.tensor(np.asarray(man_stats["rad"]), dtype=torch.float32)

    for k, v in man_stats.items():
        print(k, v.shape)

    return man_stats


###################### training with MFTMA stats ###########################

def _get_unique_gt_lb(gt_lb):
    _, idx = np.unique(gt_lb, return_index=True)
    uniq_gt_lb = gt_lb[np.sort(idx)]
    return uniq_gt_lb


def _get_dim_basis(man_stats2update, man_stats_data, dim_var_name="dim_U", known_man_dims=None):
    # dim basis need to be padded to the max dimension to be stored as a tensor

    if known_man_dims is None:
        known_man_dims = []
        for var_each_man in man_stats_data["var_explained"]:
            min_dim = np.where(var_each_man == var_each_man[var_each_man > 0.95][0])[0][0]
            known_man_dims.append(min_dim)
        known_man_dims = np.asarray(known_man_dims)
        
    dim_U = []
    padding = 0
    for i in range(known_man_dims.shape[0]):
        num_d = int(known_man_dims[i]) + 1
        if man_stats_data["var_explained"][i][num_d - 1] + 1 < 0.95:
            man_utils.print_safe(f"WARNING: {i}-th manifold has less than 95% variance ({man_stats_data['var_explained'][i][num_d - 1]})" \
                             f"explained by {num_d} dimensions.")
        
        # padding = man_stats["U"][i].shape[1] - num_d - 10

        num_d += padding # allowing for some extra dimensions
        man_utils.print_safe(f"Manifold {i} has {num_d - padding} dimensions, using {num_d} dimensions.")
        U_basis = man_stats_data["U"][i][:, :num_d]
        U_basis = np.matmul(U_basis, U_basis.T)
        dim_U.append(U_basis)  
        
    dim_U = np.array(dim_U)
    man_stats2update[dim_var_name] = torch.tensor(dim_U, dtype=torch.float32)


def load_man_stats_dict(man_stats_pth, cls2idx_mapping, orig_manifold_stats_pth = None):
    """
    load manifold statistics from pickle file
    'orig_data_var', 'centers', 'orig_global_center', 
    'corr', 'K', 'decorr_v11', 'decorr_centering_params', 'qr_q', 'cap', 'rad', 'dim', 
    'anchor_centers', 'U', 'S', 'var_explained', 'gt_lb', 'basis'
    """
    man_stats = man_utils.pickle_load(man_stats_pth)

    # gt_lb 
    gt_lb = man_stats["gt_lb"]
    man_stats["gt_lb"] = _get_unique_gt_lb(gt_lb)
    
    man_info_retriev_idx = np.zeros(len(cls2idx_mapping), dtype=int) # pos [i] stores the man index of the class man
    for i, cls_name in enumerate(man_stats["gt_lb"]):
        if cls_name in cls2idx_mapping:
            man_info_retriev_idx[cls2idx_mapping[cls_name]] = i
        else:
            man_utils.print_safe(f"WARNING: {cls_name} not found in the current training dataset class mapping!")

    man_stats["man_info_retriev_idx"] = man_info_retriev_idx

    #### transformation in model forward()
    man_stats["basis"] = torch.tensor(man_stats["basis"], dtype=torch.float32)
    man_stats["orig_global_center"] = torch.tensor(man_stats["orig_global_center"], dtype=torch.float32)
    V11 = man_stats["decorr_v11"]
    man_stats["decorr_T"] = np.identity(V11.shape[0]) - np.matmul(V11, V11.T)
    man_stats["decorr_T"] = torch.tensor(man_stats["decorr_T"], dtype=torch.float32)

    man_stats["decorr_centering_norms"] = [man_stats["decorr_centering_params"][i][1] for i in range(len(man_stats["decorr_centering_params"]))]
    man_stats["decorr_centering_norms"] = torch.tensor(man_stats["decorr_centering_norms"], dtype=torch.float32)

    man_stats["decorr_centering_means"] = np.vstack([man_stats["decorr_centering_params"][i][0] for i in range(len(man_stats["decorr_centering_params"]))])
    man_stats["decorr_centering_means"] = torch.tensor(man_stats["decorr_centering_means"], dtype=torch.float32)

    #### stored data in loss
    man_stats["centers"] = torch.tensor(np.vstack(man_stats["centers"]), dtype=torch.float32)
    man_stats["orig_data_var"] = torch.tensor(man_stats["orig_data_var"], dtype=torch.float32)
    # man_stats["orig_data_var"] /= torch.norm(man_stats["centers"], dim=1)  #TODO: normed by center norm. DO WE NEED THIS?

    man_stats["anchor_centers"] = torch.tensor(np.vstack(man_stats["anchor_centers"])[:, :-1], dtype=torch.float32)

    man_stats["rad"] = torch.tensor(man_stats["rad"], dtype=torch.float32)
    dim_mans = np.asarray(man_stats["dim"])

    # dim basis need to be padded to the max dimension to be stored as a tensor
    _get_dim_basis(man_stats, man_stats, dim_var_name="dim_U", known_man_dims=dim_mans)

    if orig_manifold_stats_pth:
        orig_man_stats = man_utils.pickle_load(orig_manifold_stats_pth)

        if type(man_stats["basis"]) != torch.Tensor:
            man_stats["basis"] = torch.tensor(man_stats["basis"], dtype=torch.float32)
        
        if type(man_stats["orig_global_center"]) != torch.Tensor:
            man_stats["orig_global_center"] = torch.tensor(man_stats["orig_global_center"], dtype=torch.float32)
        
        _get_dim_basis(man_stats, orig_man_stats, dim_var_name="orig_dim_U")
    
    return man_stats




