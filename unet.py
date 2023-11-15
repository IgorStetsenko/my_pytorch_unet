import torch

class EncodingBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodingBlock, self).__init__()
        model = [torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), torch.nn.BatchNorm2d(out_channels),
                 torch.nn.ReLU(inplace=True), torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                 torch.nn.BatchNorm2d(out_channels), torch.nn.ReLU(inplace=True)]
        self.conv = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.conv(x)


class Unet(torch.nn.Module):
    def __init__(self, out_channels=11, features=None):
        super(Unet, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = EncodingBlock(3, features[0])
        self.conv2 = EncodingBlock(features[0], features[1])
        self.conv3 = EncodingBlock(features[1], features[2])
        self.conv4 = EncodingBlock(features[2], features[3])
        self.conv5 = EncodingBlock(features[3] * 2, features[3])
        self.conv6 = EncodingBlock(features[3], features[2])
        self.conv7 = EncodingBlock(features[2], features[1])
        self.conv8 = EncodingBlock(features[1], features[0])
        self.tconv1 = torch.nn.ConvTranspose2d(features[-1] * 2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = torch.nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = torch.nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = torch.nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)
        self.bottleneck = EncodingBlock(features[3], features[3] * 2)
        self.final_layer = torch.nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x
        
        
