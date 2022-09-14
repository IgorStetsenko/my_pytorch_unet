import time
import glob

import cv2
import toml
import torch
import numpy

import albumentations

import torch.nn as nn

from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from unet import unet_model

config = toml.load('config.toml')

test = [file for file in glob.glob('data_num/data_all/test/' + '*.png')]


class encoding_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoding_block, self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*model)

    def forward(self, x):
        return self.conv(x)


class unet_model(nn.Module):
    def __init__(self, out_channels=11, features=[64, 128, 256, 512]):
        super(unet_model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = encoding_block(3, features[0])
        self.conv2 = encoding_block(features[0], features[1])
        self.conv3 = encoding_block(features[1], features[2])
        self.conv4 = encoding_block(features[2], features[3])
        self.conv5 = encoding_block(features[3] * 2, features[3])
        self.conv6 = encoding_block(features[3], features[2])
        self.conv7 = encoding_block(features[2], features[1])
        self.conv8 = encoding_block(features[1], features[0])
        self.tconv1 = nn.ConvTranspose2d(features[-1] * 2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)
        self.bottleneck = encoding_block(features[3], features[3] * 2)
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

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


class CreateDataset(Dataset):
    def __init__(self, config, transform=None):
        self.transforms = transform
        image_paths = [file for file in glob.glob(config + '*.png')]
        seg_paths = [file for file in glob.glob(config + '*.png')]
        self.images, self.masks = [], []
        for img in image_paths:
            self.images.extend([img])
        for mask in seg_paths:
            self.masks.extend([mask])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index])
        if self.transforms is not None:
            aug = self.transforms(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
            mask = torch.max(mask, dim=2)[0]
        return img, mask


def get_images(config, transform=None, batch_size=8, shuffle=True, pin_memory=True):
    data = CreateDataset(config, transform=t1)
    train_size = int(0.8 * data.__len__())
    test_size = data.__len__() - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    train_batch = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                              pin_memory=pin_memory)
    test_batch = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                                             pin_memory=pin_memory)
    return train_batch, test_batch


def FindContr(img1, n_classes):  # find the area with the number
    AnswerNumber = []
    for c in range(1, n_classes, 1):
        only_cat_hsv = cv2.inRange(img1, c, c)
        contours, hierarchy = cv2.findContours(only_cat_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # return \
        # contours of area numbers
        for cnt in contours:
            box = cv2.boundingRect(cnt)  # return (x,y) be the top-left coordinate of the rectangle \
            AnswerNumber.append(box)
            # and (w,h) be its width and height.
    return AnswerNumber


def preprocess_image(image: numpy.ndarray, device: torch.device) -> torch.Tensor:
    # Normalize image
    image = image / 255.

    # HWC to BCHW
    image = numpy.moveaxis(image, -1, 0)
    image = numpy.expand_dims(image, 0)
    # Move to device

    image = torch.from_numpy(image).to(device, non_blocking=True).float()
    image = image.resize_(11, 3, 64, 256)

    return image


def postrocess_image(image: torch.Tensor, threshold=0.5) -> numpy.ndarray:
    # Sigmoid
    image = torch.nn.Softmax()(image[0])
    # Treshold
    # image = (image > threshold).float()
    # Move to CPU
    image = image.data.cpu().numpy()
    # Normalize
    image -= image.min()
    image /= image.max()
    image *= 255.
    # Reshape image
    image = numpy.transpose(image.astype(numpy.uint8), (1, 2, 0))
    return image


if __name__ == "__main__":

    data_dir = config['file_paths']['path_to_test_image']

    t1 = albumentations.Compose([albumentations.Resize(64, 256),
                                 albumentations.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                                   std=(0.5, 0.5, 0.5)), ToTensorV2()])
    train_batch, test_batch = get_images(config, transform=t1, batch_size=8)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    best_model = unet_model().to(DEVICE)

    best_model = torch.load('steps/igor_unet_150_.pth')
    best_model.eval()

    test = [file for file in glob.glob('test3/' + '*.png')]
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for i in test:
            img0 = cv2.imread(i)

            aug = t1(image=img0)
            img0 = cv2.resize(img0, (256, 64))
            img = aug['image']
            img = img.unsqueeze(0)

            print(type(img), img.shape)
            ttim = time.time()
            x = img.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(best_model(x)), axis=1).to('cpu')
            preds1 = numpy.array(preds[0, :, :], dtype=numpy.uint8)
            print(1 / (time.time() - ttim))
            # cn = FindContr(preds1, 11)
            # for i in cn:
            #     img0 = cv2.rectangle(img0, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), (255, 255, 0), 1)
            # cv2.imshow("img", img0)
            # cv2.imshow("msk", preds1 + 100)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
