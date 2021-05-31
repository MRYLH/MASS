import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_loader import get_loader


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, num_emotions=5, repeat_num=12):
        super(Generator, self).__init__()
        c_dim = num_emotions
        layers = []
        layers.append(nn.Conv2d(1+c_dim, conv_dim, kernel_size=(3, 9), padding=(1, 4), bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.LeakyReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.LeakyReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.LeakyReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)    # 16*10*1*1
        c = c.repeat(1, 1, x.size(2), x.size(3))  # 16*10*36*256
        x = torch.cat([x, c], dim=1)              # 16*11*36*256
        return self.main(x)

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, input_size=(36, 256), conv_dim=32, repeat_num=5, num_emotions=5):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(11, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        for i in range(4):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.main = nn.Sequential(*layers)

        kernel_size_0 = int(input_size[0] / np.power(2, repeat_num))  # 1
        kernel_size_1 = int(input_size[1] / np.power(2, repeat_num))  # 8

        self.conv_dis = nn.Conv2d(curr_dim, 2, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0,
                                  bias=False)  # padding should be 0
        self.fc = nn.Linear(2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)  # 16*10*1*1    x.shape = [16,1,36,256], x1.shape = [16,11,36,256]
        c = c.repeat(1, 1, x.size(2), x.size(3))  # 16*10*36*256
        x1 = torch.cat([x, c], dim=1)  # x1.shape = [16,11,36,256]
        h1 = self.main(x1)     # [16, 512, 1, 8]
        out_src = self.conv_dis(h1)    # [16, 1, 1, 1]
        out_src = out_src.view(out_src.size(0), out_src.size(1))
        out_src = self.fc(out_src)
        out_src = self.softmax(out_src)
        return out_src

class Classifier(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, input_size=(36, 256), conv_dim=32, repeat_num=5, num_emotions=5):
        super(Classifier, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        for i in range(4):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        kernel_size_0 = int(input_size[0] / np.power(2, repeat_num))  # 1
        kernel_size_1 = int(input_size[1] / np.power(2, repeat_num))  # 8

        self.main = nn.Sequential(*layers)
        self.conv_clf_spks = nn.Conv2d(curr_dim, num_emotions, kernel_size=(kernel_size_0, kernel_size_1), stride=1,padding=0, bias=False)  # for num_speaker

        self.fc = nn.Linear(num_emotions, num_emotions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h2 = self.main(x)      # [16, 1024, 1, 8]
        out_cls_spks = self.conv_clf_spks(h2)    # [16, 10, 1, 1]
        out_cls_spks = out_cls_spks.view(out_cls_spks.size(0), out_cls_spks.size(1))
        out_cls_spks = self.fc(out_cls_spks)
        out_cls_spks = self.softmax(out_cls_spks)    # [16,num_emotions]
        return out_cls_spks


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_loader('', 16, 'train', num_workers=1)
    data_iter = iter(train_loader)
    G = Generator().to(device)
    D = Discriminator().to(device)
    C = Classifier().to(device)
    for i in range(1):
        mc_real, spk_label_org, spk_acc_c_org = next(data_iter)
        mc_real.unsqueeze_(1)     # (B, D, T) -> (B, 1, D, T) for conv2d
        mc_real = mc_real.to(device)                         # Input mc.
        spk_label_org = spk_label_org.to(device)             # Original spk labels.
        # acc_label_org = acc_label_org.to(device)             # Original acc labels.
        spk_acc_c_org = spk_acc_c_org.to(device)             # Original spk acc conditioning.
        mc_fake = G(mc_real, spk_acc_c_org)
        out_src = D(mc_fake, spk_acc_c_org)
        out_cls = C(mc_fake)



