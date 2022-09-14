import albumentations
import torch
import toml

import torch.nn as nn

from tqdm import tqdm
from unet import Unet
from torchsummary import summary
from custom_dataloader import get_images

from albumentations.pytorch import ToTensorV2
from torch.optim import Adam

config = toml.load('config.toml')

LEARNING_RATE = config['neural_network_settings']['LEARNING_RATE']
NUM_EPOCHS = config['neural_network_settings']['NUM_EPOCHS']
DEVICE = config['neural_network_settings']['DEVICE']
BATCH_SIZE = config['neural_network_settings']['BATCH_SIZE']
H, W = config['neural_network_settings']['HEIGHT'], config['neural_network_settings']['WEIGHT']

model = Unet().to(DEVICE)

if __name__ == "__main__":

    t1 = albumentations.Compose([
        albumentations.Resize(H, W),
        albumentations.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()])
    train_batch, test_batch = get_images(config, transform=t1, batch_size=BATCH_SIZE)

    summary(model, (3, H, W))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    number_epoch = 0
    for epoch in range(NUM_EPOCHS):
        print("epoch___" + str(number_epoch))
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
        number_epoch += 1
        if number_epoch % 10 == 0:
            torch.save(model, 'steps/igor_unet_' + str(number_epoch) + '_.pth')
    torch.save(model, 'weights/igor_unet_' + str(number_epoch) + '_.pth')
