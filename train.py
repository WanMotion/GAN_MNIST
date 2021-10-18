import torchvision.datasets
from torch.utils.data import DataLoader
from config import *
from RecogNet import RecogNet
from GenerateNet import GenerateNet
from loss import *
import os
import time


def process_img(X):
    trans = torchvision.transforms.ToTensor()
    return (trans(X) - 0.5) / 0.5


def deprocess_img(X):
    return (X + 1.0) / 2.0

# 选择设备
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# 数据集
MnistDataSet = torchvision.datasets.MNIST("dataset", train=True, download=True, transform=process_img)
dataLoader = DataLoader(MnistDataSet, batch_size=64, shuffle=True)

# 网络
DNet = RecogNet()
DNet.to(device)
DNet.train()
GNet = GenerateNet(NOISE_DIM)
GNet.to(device)
GNet.train()

# 优化器
DOptimizer = torch.optim.SGD(DNet.parameters(), lr=0.001, momentum=1e-5)
torch.optim.lr_scheduler.StepLR(DOptimizer,100,0.5,-1)  # 等间距调整学习率
GOptimizer = torch.optim.SGD(GNet.parameters(), lr=0.001, momentum=1e-5)
torch.optim.lr_scheduler.StepLR(GOptimizer,100,0.5,-1)

# 损失函数
DLoss = DiscriminatorLoss()
DLoss.to(device)
GLoss = GeneratorLoss()
GLoss.to(device)

# train
for i in range(EPOCH):
    gLoss = 0
    for x, label in dataLoader:
        batchSize = x.shape[0]
        x.to(device)
        # 先训练判别器
        for k in range(5):
            d_out = DNet(x)
            # 固定生成器，根据噪音生成
            noise = torch.autograd.Variable(torch.randn(batchSize, NOISE_DIM)).to(device)
            fakeData = GNet(noise)
            g_out = DNet(fakeData)
            ls = DLoss(d_out, g_out)
            DOptimizer.zero_grad()
            ls.backward()
            DOptimizer.step()
        # 再固定判别器，训练生成器
        noise = torch.autograd.Variable(torch.randn(batchSize, NOISE_DIM)).to(device)
        # 生成fake数据
        fakeData = GNet(noise)
        # 放入DNet判别
        d_out = DNet(fakeData)
        # 送入损失函数
        g_loss = GLoss(d_out)
        # 优化
        GOptimizer.zero_grad()
        g_loss.backward()
        GOptimizer.step()
        gLoss += g_loss

    if i % 100 == 0:
        if i > 0:
            torch.save(GNet.state_dict(), os.path.join(WEIGHTS_OUTPUT_PATH, f"{round(time.time())}epoch_{i + 1}.pth"))
            noise = torch.randn(9, NOISE_DIM)
            fakeData = GNet(noise)
            toImage=torchvision.transforms.ToPILImage()
            for k in range(9):
                image=toImage(fakeData[k].cpu().clone().squeeze(0))
                image.save(os.path.join(WEIGHTS_OUTPUT_PATH,f"iter_{i}_{k}.jpg"))

        print(f"iter:{i},g_loss:{GLoss}")
