import torch.utils.data as data

from utils.img_loader import default_loader


class ImageNetDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, loader=default_loader):
        self._data_dir = data_dir
        self._transform = transform
        self._loader = loader

    def __getitem__(self, index):
        pass
