import torch.autograd
from torch import nn

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.loss=nn.BCEWithLogitsLoss()

    def forward(self,logitsReal,logitsFake):
        size=logitsReal.shape[0]
        targetReal=torch.autograd.Variable(torch.ones(size,dtype=torch.float)).view(-1,1)
        targetFake=torch.autograd.Variable(torch.zeros(size,dtype=torch.float)).view(-1,1)
        return self.loss(logitsReal,targetReal)+self.loss(logitsFake,targetFake)

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.loss=nn.BCEWithLogitsLoss()

    def forward(self,logitsFake):
        size=logitsFake.shape[0]
        target=torch.autograd.Variable(torch.zeros(size,dtype=torch.float)).view(-1,1)
        return self.loss(logitsFake,target)