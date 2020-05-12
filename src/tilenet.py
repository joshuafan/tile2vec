'''Modified ResNet-18 in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TileNet(nn.Module):
    def __init__(self, num_blocks, in_channels=4, z_dim=512):
        super(TileNet, self).__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.in_planes = 20

        self.conv1 = nn.Conv2d(self.in_channels, self.in_planes, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(20)
        self.layer1 = self._make_layer(20, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(64, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, num_blocks[3], stride=1)
        self.layer5 = self._make_layer(self.z_dim, num_blocks[4],
            stride=2)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.z_dim, self.z_dim)  # TODO change
        self.decoder1 = nn.ConvTranspose2d(self.z_dim, 128, 3, stride=1)
        self.decoder2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)
        self.decoder3 = nn.ConvTranspose2d(64, self.in_channels, 2, stride=2, padding=0)


    def _make_layer(self, planes, num_blocks, stride, no_relu=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_planes, planes, stride=stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(self.conv1(x))
        x = self.layer1(x)
        #print('after layer 1', x.shape)
        x = self.layer2(x)
        #print('after layer 2', x.shape)
        x = self.layer3(x)
        #print('after layer 3', x.shape)
        x = self.layer4(x)
        #print('after layer 4', x.shape)
        x = self.layer5(x)
        #print('after layer 5', x.shape)
        x = F.avg_pool2d(x, 2)
        #print('after avgpool', x.shape)
        z = x.view(x.size(0), -1)
        #print('z', z.shape)
        
        #x = self.avgpool(x)
        #print('after avgpool', x.shape)
        # x = torch.flatten(x, 1)
        ##print('after flatten', x.shape)
        z = self.fc(z)
        #print('final embedding:', z.shape)
        return z

    def decode(self, z):
        #print('DECODING! Z dim', z.shape)
        x = z[:, :, None, None]
        #print('Reshaped', x.shape)
        x = F.relu(self.decoder1(x))
        #print('After decoder1', x.shape)
        x = F.relu(self.decoder2(x))
        #print('After decoder2', x.shape)
        x = self.decoder3(x)
        #print('After decoder3', x.shape)
        return x

    def forward(self, x):
        return self.encode(x)

    def triplet_loss(self, z_p, z_n, z_d, margin=0.1, l2=0):
        l_n = torch.sqrt(((z_p - z_n) ** 2).sum(dim=1))
        l_d = - torch.sqrt(((z_p - z_d) ** 2).sum(dim=1))
        l_nd = l_n + l_d
        loss = F.relu(l_n + l_d + margin)
        l_n = torch.mean(l_n)
        l_d = torch.mean(l_d)
        l_nd = torch.mean(l_n + l_d)
        loss = torch.mean(loss)
        if l2 != 0:
            loss += l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
        return loss, l_n, l_d, l_nd

    def loss(self, patch, neighbor, distant, margin=0.1, l2=0):
        """
        Computes loss for each batch.
        """
        #print('patch shape', patch.shape)
        z_p, z_n, z_d = (self.encode(patch), self.encode(neighbor),
            self.encode(distant))
        #print('Embedding', z_p)

        # Compute reconstruction loss
        recon_p, recon_n, recon_d = (self.decode(z_p), self.decode(z_n),
                self.decode(z_d))
        #print('Band means of original patch', patch.mean(dim=(2,3))[0])
        #print('Band means of reconstructed', recon_p.mean(dim=(2,3))[0])
        criterion = nn.MSELoss()
        l_recon_p = criterion(patch, recon_p)
        l_recon_n = criterion(neighbor, recon_n)
        l_recon_d = criterion(distant, recon_d)
        l_recon = (l_recon_p + l_recon_n + l_recon_d) / 3
        #print('l_recon', l_recon)

        RECON_WEIGHT = 1.0 #0.5

        # Compute triplet loss
        loss, l_n, l_d, l_nd = self.triplet_loss(z_p, z_n, z_d, margin=margin, l2=l2)
        loss += (l_recon * RECON_WEIGHT)
        return loss, l_n, l_d, l_nd, l_recon

def make_tilenet(in_channels=4, z_dim=512):
    """
    Returns a TileNet for unsupervised Tile2Vec with the specified number of
    input channels and feature dimension.
    """
    num_blocks = [2, 2, 2, 2, 2]
    return TileNet(num_blocks, in_channels=in_channels, z_dim=z_dim)

