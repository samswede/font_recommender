
#%%
from train import *

""" TO DO:
    - VGG16 expects 3 channels, not grayscale.
        Either need a new model or find another solution. 
        Probably new model.

"""


config = {
        'dataset_path': '/Users/samuelandersson/Dev/github_projects/font_recommender/data/fonts/Aa_improved',

        'latent_dims': 9,
        'num_epochs': 20,
        'learning_rate': 0.00005,
        'weight_decay': 1e-5,
        'model_size': 'big', 

        'model_save_folder_path': '',
        'vgg16_model_path': './models/vgg16.pth',

        'model_save_epoch_interval': 100,
        'print_performance_epoch_interval': 10
    }

main(config)
# %%
from NEW_variational_autoencoder import *

vgg16_model_path = './models/vgg16.pth'
perceptual_loss = PerceptualLoss(vgg16_model_path)
# %%

vgg16 = torch.load("./models/vgg16.pth")

# %%
import torch
from torchvision import models

# Initialize the model
vgg16_model = models.vgg16()

# Load the state_dict from the file
vgg16_state_dict = torch.load("./models/vgg16.pth")

# Update the model's state_dict
vgg16_model.load_state_dict(vgg16_state_dict)


# %%
