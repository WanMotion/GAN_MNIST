from torch import nn


class RecogNet(nn.Module):
    def __init__(self):
        super(RecogNet, self).__init__()
        self.convNet = nn.Sequential(
            nn.Conv2d(1, 12, (3, 3), (1, 1), padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d((2, 2), (2, 2)),  # (14,14)
            nn.Conv2d(12, 36, (3, 3), (1, 1), padding=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d((2, 2), (2, 2)),  # (7,7)
        )
        self.fcNet = nn.Sequential(
            nn.Linear(7 * 7 * 36, 7 * 7 * 36),
            nn.LeakyReLU(0.01),
            nn.Linear(7 * 7 * 36, 1)  # 只需要输出是否是数字的置信度，不需要区分数字类别
        )

    def forward(self, x):
        conv = self.convNet(x)
        fc = self.fcNet(conv.view(x.shape[0], -1))
        return fc
