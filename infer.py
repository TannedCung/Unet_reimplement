import torch
from torchvision import transforms
from PIL import Image
from network.network import Unet
import torch.nn as nn
import torch.optim as optim
from utils.preprocessing import *
from utils.postprocessing import *
import numpy as np
import cv2

class inference():
    def __init__(self):
        path = "checkpoint/model_699.pth"
        self.net = Unet()
        self.net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.net.eval()
        self.pre = Preprocess()
        self.post = postProcess(size=512, n_class=2)

    def infer(self, img):
        img = self.pre.Pre(img)
        out = self.net(img)
        out = self.post.post(out)
        return out


Checker = inference()
path = './data/test/test-volume-0.png'
img = Image.open(path)
# img = img.convert('1')
img.show()
out = Checker.infer(img)
out1 = Image.fromarray((out[0] * 255).astype(np.uint8))
out2 = Image.fromarray((out[1] * 255).astype(np.uint8))
out1.show()
out2.show()
