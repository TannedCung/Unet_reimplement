import torch
from torchvision import transforms
from PIL import Image

class Preprocess():
    def __init__(self):
        self.transform = transforms.Compose([
                        transforms.Resize(572),
                        transforms.ToTensor()
        ])
    def Pre(self, img):
        img = self.transform(img)
        img = img.unsqueeze(0)
        return img
