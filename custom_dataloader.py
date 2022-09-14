import glob
import torch
import toml
import cv2

config = toml.load('config.toml')

class LoadImages:
    def __init__(self, config, transform=None):
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
    data = LoadImages(config, transform=transform)
    train_size = int(0.8 * data.__len__())
    test_size = data.__len__() - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    train_batch = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                              pin_memory=pin_memory)
    test_batch = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                                             pin_memory=pin_memory)
    return train_batch, test_batch