from torchvision import transforms
from torch.utils.data import Dataset


class CustomDataloader(Dataset):
    # mean, std calculated from the training split of MSCOCO
    TRAIN_MEAN = [0.4740539, 0.45110828, 0.4117162]
    TRAIN_STD = [0.1120, 0.1153, 0.1304]

    IMG_TF = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
    ])

    def __init__(self, input_img_data, voxel_data):
        self.voxel_data = voxel_data
        self.img_data = input_img_data
        self.transform = self.IMG_TF

    def __len__(self):
        return self.img_data.shape[0]

    def __getitem__(self, idx):
        img = self.img_data[idx].transpose(1, 2, 0)
        image = self.transform(img)
        neural_activity = self.voxel_data[idx, :]
        return image, neural_activity
