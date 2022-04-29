import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='valid'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='valid'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='valid'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            nn.Flatten(),
            nn.Linear(in_features=10 * 10 * 64, out_features=512),
            nn.Linear(in_features=512, out_features=1),
        )

    def forward(self, X):
        return self.network(X)