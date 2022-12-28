import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.linear1 = nn.Linear(1152, 64)
        self.dropout = nn.Dropout(0.25)

        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25)
        self.output = nn.Linear(32 * 2, num_chars + 1)  # bidirectional

    def forward(self, images, targets=None):
        b, c, h, w = images.size()

        x = F.relu(self.conv1(images))
        x = F.relu(self.maxpool1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.maxpool2(x))  # 1, 64, 18, 75
        x = x.permute(0, 3, 1, 2)  # 1, 75, 64, 18
        x = x.view(b, x.size(1), -1)

        x = self.dropout(self.linear1(x))

        x, _ = self.gru(x)
        x = self.output(x)  # 1, 75, 20 - 75 time steps, 20 characters

        x = x.permute(1, 0, 2)  # timestep, bs, values

        if targets is not None:
            log_softmax_values = F.log_softmax(
                x, dim=2
            )  # dim is dimension where classes are
            print(log_softmax_values.size())
            # input_length = torch.full()

        return x


if __name__ == "__main__":
    img = torch.rand(1, 3, 75, 300)
    model = CaptchaModel(19)
    target = torch.randint(1, 20, (1, 5))
    x = model(img, target)

    print(x.size())
