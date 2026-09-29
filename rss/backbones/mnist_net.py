import torch

class MNIST_Net(torch.nn.Module):
    def __init__(self, N=10, channels=1):
        super(MNIST_Net, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 12, 5),
            torch.nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            torch.nn.ReLU(True),
            torch.nn.Conv2d(12, 16, 5),  # 6 12 12 -> 16 8 8
            torch.nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            torch.nn.ReLU(True)
        )
        self.classifier_mid = torch.nn.Sequential(
            torch.nn.Linear(16 * 4 * 4, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU())
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(84, N),
            torch.nn.Softmax(1)
        )
        self.channels = channels

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d):
            print('init conv2, ', m)
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

        if isinstance(m, torch.nn.Linear):
            print('init Linear, ', m)
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier_mid(x)
        x1 = self.classifier(x)
        return x1, x, x