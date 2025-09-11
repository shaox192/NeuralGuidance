import os
import torch
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import Dataset
import numpy as np
from PIL import Image


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImageFolderSelected:

    def __init__(self, folder_root, folder_ls, transform, shuffle=False):
        self.root = folder_root
        self.folder_ls = folder_ls
        self.shuffle = shuffle

        self.transform = transform
        classes, class_to_idx = self.find_classes()
        self.samples = self.make_dataset(class_to_idx)

        self.loader = pil_loader

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in self.samples]

        rng = np.random.default_rng(1024)
        shuffle_idx = rng.choice(np.arange(len(self.samples)), len(self.samples), replace=False)
        self.idx2shuffledidx = {i: shuffle_idx[i] for i in range(len(shuffle_idx))}

    def find_classes(self):
        classes = sorted(self.folder_ls)  # !! alphabetical!
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

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
        if self.shuffle:
            idx = self.idx2shuffledidx[index]
            shuffled_path, _ = self.samples[idx]
            shuffled_img = self._load_img(shuffled_path)
        else:
            shuffled_img = 0

        return img, target, shuffled_img

    def __len__(self) -> int:
        return len(self.samples)


def load_data_folder(prt_data_pth, folder_names, distributed, batch_size,
                     train_workers, test_workers,
                     shuffle=False):
    """
    load data using ImageFolder for selected folders (categories)
    :param args:
    :return:
    """

    traindir = os.path.join(prt_data_pth, 'train')
    valdir = os.path.join(prt_data_pth, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ImageFolderSelected(
        traindir,
        folder_names,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        shuffle=shuffle)

    val_dataset = ImageFolderSelected(
        valdir,
        folder_names,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

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
