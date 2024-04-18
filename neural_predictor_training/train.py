import argparse
from os.path import join as join_pth

import h5py
import numpy as np

import time

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.models as models

from CustomDataloader import CustomDataloader
import utils


parser = argparse.ArgumentParser(description='train neural predictor with already averaged and pcaed data')
parser.add_argument('--sub', default='sub1', type=str, help='sub1, sub2, ...')
parser.add_argument('--roi', default='V1', type=str,
                    help='roi name: [V1], [hV4]...')

parser.add_argument('--data-dir', type=str, help='where to find the training data')
parser.add_argument('--save-dir', type=str, help='where to save outputs')

parser.add_argument('--shuffle', action='store_true', help="shuffle neural data during training - control condition")

parser.add_argument('--lr',  default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epk', default=40, type=int, help='number of epochs', dest='epk')
parser.add_argument('--save-interval', default=5, type=int, help='when to save checkpoint', dest='epk')


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Using device: {DEVICE} <--")


class VoxelLoss(nn.Module):
    def __init__(self, floor=0.1):
        super(VoxelLoss, self).__init__()
        self.weight_floor = floor

    def forward(self, predicted_repr, target_repr):
        batch_size = predicted_repr.shape[0]
        err = torch.sum((predicted_repr - target_repr)**2, dim=0) # voxel pattern for each image vs. predicted
        loss = torch.sum(err)

        loss /= batch_size
        return loss


def data_loader(img, voxel, batch_size, shuffle):
    data = CustomDataloader(img, voxel)
    data_ldr = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_ldr


def train(data_loader, model, criterion, optimizer, val_loader, num_epochs=10, save_dir_name="", save_interval=5):
    train_loss_per_epoch = []
    val_corr_per_epoch = []
    val_loss_per_epoch = []
    for epoch in range(num_epochs):
        tic = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print(f"Using learning rate: {optimizer.param_groups[0]['lr']}")
        print('-' * 10)

        model.train()  # Set model to training mode

        running_loss = 0.0
        cnt = 0
        # Iterate over data.
        for inputs, target_outputs in data_loader:
            inputs, target_outputs = inputs.to(DEVICE), target_outputs.to(DEVICE)
            vox_size = target_outputs.shape[1]
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, target_outputs)

            loss.backward()

            optimizer.step()
            running_loss += (loss.item()/vox_size)
            cnt += 1
        epoch_loss = running_loss / cnt
        train_loss_per_epoch.append(epoch_loss)
        print(f"* Loss: {epoch_loss}")
        print(f"* current epoch takes: {time.time() - tic}", flush=True)

        val_corr, val_loss = test(val_loader, model, criterion)
        val_corr_per_epoch.append(val_corr)
        val_loss_per_epoch.append(val_loss)

        if epoch % save_interval == 0 and epoch != 0:
            torch.save(model, f"{save_dir_name}_epoch_{epoch}.pth")

    return model, train_loss_per_epoch, val_corr_per_epoch, val_loss_per_epoch


def test(data_ldr, model, criterion):
    model.eval()
    val_data_size = len(data_ldr.dataset)
    corr_full = np.zeros(val_data_size)

    full_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, data in enumerate(data_ldr):
            inputs, target_outputs = data

            batch_size = target_outputs.shape[0]
            vox_size = target_outputs.shape[1]
            inputs, target_outputs = inputs.to(DEVICE), target_outputs.to(DEVICE)

            ## use these with regular resnet, alexnet...
            predicted_outputs = model(inputs)
            loss = criterion(predicted_outputs, target_outputs)

            full_loss += (loss.item()/vox_size)

            corrs = []
            for i in range(target_outputs.shape[0]):
                c = np.corrcoef(target_outputs[i, :].cpu().numpy(), predicted_outputs[i, :].cpu().numpy())[0, 1]
                corrs.append(c)
            corr_full[count: count + batch_size] = corrs
            count += batch_size

    full_loss /= val_data_size
    max_corr = np.max(corr_full)
    mean_corr = np.mean(corr_full)
    median_corr = np.median(corr_full)
    print(f"Testing on validation set, loss: {full_loss}, max correlation: {max_corr}, mean correlation: {mean_corr}, "
          f"median correlation: {median_corr}", flush=True)

    return corr_full, full_loss


def shuffle_img_id(val_id, train_id):
    val_size, train_size = val_id.shape[0], train_id.shape[0]
    full_id = np.append(val_id, train_id)

    np.random.seed(1024)
    full_id = np.random.choice(full_id, full_id.shape[0], replace=False)

    return full_id[:val_size], full_id[val_size:]


def main(args):

    # save stuff
    save_dir = join_pth(args.save_dir, f"{args.sub}_np_ckpt")
    save_prefix = join_pth(save_dir, f"np_{args.roi}_shuffle") if args.shuffle else \
                    join_pth(save_dir, f"np_{args.roi}")
    utils.make_directory(save_dir)

    # neural data:
    beta_f = join_pth(args.data_dir, f"{args.sub}_{args.roi}_data.pkl")
    order_data_orig = utils.pickle_load(beta_f)

    val_img_id = order_data_orig["val_imgID"]
    val_beta = order_data_orig["val_beta"]
    train_img_id = order_data_orig["train_imgID"]
    train_beta = order_data_orig["train_beta"]

    if args.shuffle:
        print(f"\n\t=======> Shuffling!! <=======\n", flush=True)
        val_img_id, train_img_id = shuffle_img_id(val_img_id, train_img_id)

    print(f"-> Loaded order and neural data from {beta_f}...\n"
          f"\tval image ID: {val_img_id.shape}, val beta: {val_beta.shape}",
          f"\ttrain image ID: {train_img_id.shape}, val beta: {train_beta.shape}", flush=True)

    # stimuli images
    stim_f = join_pth(args.data_dir, f"{args.sub}_stimuli_227.h5py")
    image_data_h5py = h5py.File(stim_f, 'r')
    image_data = np.copy(image_data_h5py['stimuli'])
    image_data_h5py.close()
    print(f"*** Loaded stim images from {stim_f}, image data shape: {image_data.shape}")

    # extract validation set
    val_images = image_data[val_img_id]
    print(f"*** Validation set, image shape: {val_images.shape}")
    val_data_ldr = data_loader(val_images, val_beta, 100, False)

    # load training set
    train_images = image_data[train_img_id]
    print(f"*** Training set, image shape: {train_images.shape}")
    train_data_ldr = data_loader(train_images, train_beta, 200, True)

    # initialize model, loss and optimizer
    net = models.resnet18(num_classes=train_beta.shape[1])
    print(f"\n******* Initialized neural net:\n {net.__repr__()}\n*******")
    net.to(DEVICE)

    criterion = VoxelLoss()
    print("\n*** Using Adam optimizer")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)

    # training
    train(train_data_ldr, net,criterion, optimizer, val_data_ldr, args.epk, save_prefix, args.save_interval)

    print("DONE.", flush=True)


if __name__ == "__main__":
    args = parser.parse_args()
    utils.show_input_args(args)

    main(args)

