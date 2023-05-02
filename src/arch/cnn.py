import torch.nn as nn

from utils import model_init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=512):
        super(CNNBase, self).__init__()
        self.hidden_size = hidden_size

        init_ = lambda m: model_init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        self.cnn_base = nn.Sequential(
            init_(
                nn.Conv2d(
                    in_channels=num_inputs,
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                )
            ),
            nn.ReLU(),
            init_(
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2
                )
            ),
            nn.ReLU(),
            init_(
                nn.Conv2d(
                    in_channels=64, out_channels=32, kernel_size=3, stride=1
                )
            ),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU(),
        )

        self.train()

    def forward(self, inputs):
        x = self.cnn_base(inputs / 255.0)
        return x
