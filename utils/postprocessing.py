import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F 

class postProcess():
    def __init__(self, size, n_class):
        self.size = size
        self.n_class = n_class
        self.transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(self.size),
                        transforms.ToTensor()
                        ])
    
    def post(self, output):
        if self.n_class > 1:
            out = F.softmax(output, dim=1)
        else:
            out = torch.sigmoid(output)
        out = out.squeeze(0)
        out = self.transform(out.cpu())
        out = out.squeeze().cpu().numpy()
        return out

