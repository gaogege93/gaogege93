import torch


# 二维互相关运
def corr2d(X, K):
    h, w = K.shape
    y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (X[i:i + h, j:j + w] * K).sum().item()
    return y


# 二维卷积层
class Conv2d(torch.nn.Module):
    def __init__(self, kernel_size):
        super(Conv2d, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


x = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
print(x)
net = torch.nn.Conv2d(in_channels=1, out_channels=1, padding=2, stride=2, kernel_size=2)
print(net(x))
