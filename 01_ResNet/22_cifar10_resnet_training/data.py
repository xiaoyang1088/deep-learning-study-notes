import torch 
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

def create_data_loaders(data_dir,batch_size=128):
    train_dataset=datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=False
    )
    test_dataset=datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=False
    )

    mean_image=torch.tensor(
        train_dataset.data.mean(axis=0),
        dtype=torch.float32
    ).permute(2,0,1)    #调整维度顺序

    def subtract_mean(image):
        return image.float()-mean_image

    train_dataset.transform=transforms.Compose([
        transforms.PILToTensor(),
        subtract_mean,
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    test_dataset.transform=transforms.Compose([
        transforms.PILToTensor(),
        subtract_mean
    ])

    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    test_loader=DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    return train_loader,test_loader