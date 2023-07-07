
#%%
from train import *

config = {
        'dataset_path': '/Users/samuelandersson/Dev/github_projects/font_recommender/data/fonts/Aa_improved',

        'latent_dims': 9,
        'num_epochs': 5,
        'learning_rate': 0.00005,
        'weight_decay': 1e-5,
        'model_size': 'test', 

        'model_save_folder_path': './models',
        'vgg16_model_path': './models/vgg16.pth',

        'model_save_epoch_interval': 1,
        'print_performance_epoch_interval': 1
    }

main(config)


# %%
