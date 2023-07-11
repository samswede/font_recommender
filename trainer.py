import torch
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader,random_split


class Trainer():
    def __init__(self, dataset_path, device, batch_size = 32):

        self.batch_size = batch_size
        self.image_size = (128, 128)
        
        #split = [int(m*0.9), int(m*0.1)]
        split = [850, 65]   # hard coded split (total 915 in dataset)
        self.prepare_data_loaders(dataset_path, split)

        self.device = device
        self.size_dataset = len(self.train_loader.dataset)

        
        
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def set_image_size(self, image_size):
        self.image_size = image_size

    def train_epoch(self, model, beta= 1):
        """
        """
        train_loss = 0.0

        # Create weighting term to balance KL-divergence loss
        #beta = torch.tensor(beta).type(model.encoder.kl.dtype).to(model.encoder.kl.device)
        #beta = torch.tensor(beta).to(model.encoder.kl.device)
        beta = torch.tensor(beta).to(model.encoder.device)

        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for x, _ in self.train_loader: 
            # Move tensor to the proper device
            x = x.to(self.device)
            # Forward pass and Backpropagation
            loss = model.back_prop(x, beta)
            # Accumulate the batch loss
            train_loss += loss.item()
        # Return the average loss
        return train_loss / self.size_dataset
    
    def test_epoch(self, model, beta= 1):
        """
        """
        val_loss = 0.0

        # Create weighting term to balance KL-divergence loss
        beta = torch.tensor(beta).type(model.encoder.kl.dtype).to(model.encoder.kl.device)

        with torch.no_grad(): # No need to track the gradients
            for x, _ in self.test_loader:
                # Move tensor to the proper device
                x = x.to(self.device)
                # Forward pass through the VAE
                x_hat = model.forward(x)

                # Compute the loss
                loss = model.perceptual_loss(x, x_hat) + beta * model.encoder.kl
                # Accumulate the batch loss
                val_loss += loss.item()
        # Return the average loss
        return val_loss / self.size_dataset

    def get_mean_and_std(self, loader):
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
    
    def prepare_initial_data_loader(self, dataset_path):

        # Initial transform
        initial_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1)
        ])

        # Define initial train & test dataset
        initial_train_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=initial_transform)

        # Create initial loaders
        initial_train_loader = torch.utils.data.DataLoader(initial_train_dataset, batch_size=self.batch_size, shuffle=True)
        return initial_train_loader

    def prepare_data_loaders(self, dataset_path, split):

        #split = [int(m*0.9), int(m*0.1)]
        self.split = split

        initial_train_loader = self.prepare_initial_data_loader(dataset_path)

        # Compute mean and standard deviation
        mean, std = self.get_mean_and_std(initial_train_loader)

        print(f'mean: {mean}')
        print(f'std: {std}')

        # Transform with normalization
        self.transform_norm = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Grayscale(num_output_channels=1)
        ])

        # Redefine train & test dataset with normalization
        self.train_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=self.transform_norm)
        self.test_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=self.transform_norm)

        m = len(self.train_dataset)
        print(f'length dataset: {m}')
        
        self.train_data, self.val_data = random_split(self.train_dataset, self.split)

        # The dataloaders handle shuffling, batching, etc...
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)


