from torch import nn


class GenerateNet(nn.Module):
    def __init__(self, noiseDim: int):
        super(GenerateNet, self).__init__()
        self.noiseDim = noiseDim
        self.net1 = nn.Sequential(
            nn.Linear(self.noiseDim, 1024),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(7 * 7 * 128)
        )
        self.net2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,1,(4,4),(2,2),(1,1)),
            nn.LeakyReLU(0.01)
        )

    def forward(self,x):
        net1=self.net1(x)
        net2=self.net2(net1.view(x.shape[0],128,7,7))
        return net2
