import torch
from torch.autograd import Variable

class LinearRegression(torch.nn.Module):
    # response_curves
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x.transpose(1,0))
        return out.view(-1)
    