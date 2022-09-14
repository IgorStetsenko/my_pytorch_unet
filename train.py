from torch.utils.data import Dataset
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A
import glob

import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import cv2
from unet import unet_model
from torchsummary import summary





class LyftUdacity(Dataset):
    def __init__(self, img_dir, transform=None):
        self.transforms = transform
        image_paths = [file for file in glob.glob('data1/images/' + '*.png')]
        seg_paths = [file for file in glob.glob('data1/masks/' + '*.png')]
        self.images, self.masks = [], []
        for i in image_paths:
            imgs = i
            self.images.extend([i])
        for i in seg_paths:
            masks = i

            self.masks.extend([i])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            aug = self.transforms(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
            mask = torch.max(mask, dim=2)[0]
        return img, mask


def get_images(image_dir, transform=None, batch_size=1, shuffle=True, pin_memory=True):
    data = LyftUdacity(image_dir, transform=t1)
    train_size = int(0.8 * data.__len__())
    test_size = data.__len__() - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    train_batch = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                              pin_memory=pin_memory)
    test_batch = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                                             pin_memory=pin_memory)
    return train_batch, test_batch


def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)), axis=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()


if __name__ == "__main__":
    data_dir = 'data_num/daes/'
    t1 = A.Compose([
        A.Resize(64, 256),
        A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    train_batch, test_batch = get_images(data_dir, transform=t1, batch_size=8)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = unet_model().to(DEVICE)

    summary(model, (3, 256, 256))

    LEARNING_RATE = 1e-4
    num_epochs = 400
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    i = 0
    for epoch in range(num_epochs):
        print("epoch___" + str(i))
        loop = tqdm(enumerate(train_batch), total=len(train_batch))
        for batch_idx, (data, targets) in loop:
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            targets = targets.type(torch.long)
            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
        i += 1
        if i % 10 == 0:
            torch.save(model, 'steps/igor_unet_' + str(i) + '_.pth')
