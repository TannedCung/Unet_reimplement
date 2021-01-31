import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image

class UnetDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.train_path = os.path.join(path, "train")
        self.filenames = list(glob.iglob(os.path.join(self.train_path,"*.*")))
        self.label_path = os.path.join(path, "label")
        self.transform = T.Compose([T.Resize(572),
                                            T.ToTensor()])
        self.transform_1 = T.Compose([T.Resize(508),
                                            T.ToTensor()])

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        X_path = os.path.join(self.train_path, "train-volume-{}.png".format(idx))
        Y_path = os.path.join(self.label_path, "train-labels-{}.png".format(idx))    
        X = Image.open(X_path)
        Y = Image.open(Y_path)
        X = self.transform(X)
        # X = X.unsqueeze(0)
        Y = self.transform_1(Y)
        Y = Y.squeeze(0)
        # Y = torch.tensor(Y, dtype=torch.long)
        data = [X, Y]
        return data