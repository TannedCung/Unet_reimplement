from PIL import Image
import cv2
import torch
from network.network import Unet
from datasets.dataloader import UnetDataset
import torch.optim as optim
import torch.nn as nn

# criterion = nn.CrossEntropyLoss()
criterion = nn.SmoothL1Loss()
save_path = "checkpoint/model.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Net = Unet()
Net.load_state_dict(torch.load(save_path))
Net.to(device)
Net.train()
opt = optim.Adam(Net.parameters(), lr=0.001)
Uta = UnetDataset(path="./data")
data = torch.utils.data.DataLoader(Uta, batch_size=3)
print("init net done")

for epoch in range(15):
    running_loss = 0
    for i, d in enumerate(data):
        [X, Y] = d[0].to(device), d[1].to(device)

        opt.zero_grad()

        out = Net(X)
        out1 = out[:,0:1,:,:]
        out2 = out[:,1:2,:,:]
        loss = criterion(out, Y)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print("[{}, {}], loss {:.3}".format(epoch+1, i+1, running_loss/10))
        running_loss = 0.0
    torch.save(Net.state_dict(), save_path)
    print("model saved to{}".format(save_path))




