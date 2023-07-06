import torch
from torchvision import transforms
from torch.utils.data import DataLoader,random_split

from utils import *


def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_a_batch

    mean /= total_images_count
    std /= total_images_count

    return mean, std

def prepare_data_loaders(dataset_path, image_size=(128, 128), batch_size=32):
    # Initial transform
    initial_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])

    # Define initial train & test dataset
    train_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=initial_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=initial_transform)

    # Create initial loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Compute mean and standard deviation
    mean, std = get_mean_and_std(train_loader)

    print(f'mean: {mean}')
    print(f'std: {std}')

    # Transform with normalization
    transform_norm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Grayscale(num_output_channels=1)
    ])

    # Redefine train & test dataset with normalization
    train_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform_norm)
    test_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform_norm)

    m = len(train_dataset)
    print(f'length dataset: {m}')
    #split = [int(m*0.9), int(m*0.1)]
    split = [850, 65]
    train_data, val_data = random_split(train_dataset, split)

    # The dataloaders handle shuffling, batching, etc...
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader, train_dataset, test_dataset, transform_norm
