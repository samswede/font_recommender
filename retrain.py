import torch
from utils import *
from NEW_variational_autoencoder import *
from trainer import Trainer
import re  # for extracting epoch number from file name

# Define the naming convention
def model_file_naming_convention(model_size, latent_dims, epoch):
    model_file_name = f'VAE_S{model_size}_L{latent_dims}_E{epoch}.pt'
    return model_file_name

# Define the model saving function
def save_model(model, model_file_name, model_save_folder_path):
    model_path = F'{model_save_folder_path}/{model_file_name}'
    torch.save(model.state_dict(), model_path)

# Define the model loading function
def load_model(model, model_file_name, model_save_folder_path):
    model_path = F'{model_save_folder_path}/{model_file_name}'
    model.load_state_dict(torch.load(model_path))

# Define the main function
def main(config):

    # Unpack arguments
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_epochs = config['num_epochs']

    model_size = config['model_size']
    latent_dims = config['latent_dims']

    dataset_path = config['dataset_path']
    model_save_folder_path = config['model_save_folder_path']
    vgg16_model_path = config['vgg16_model_path']

    pretrained_model_file_name = config['pretrained_model_file_name']

    print_performance_epoch_interval = config['print_performance_epoch_interval']
    model_save_epoch_interval = config['model_save_epoch_interval']

    # Set replicable random seed
    torch.manual_seed(0)

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Instantiate classes and load pretrained model
    vae = VariationalAutoencoder(latent_dims= latent_dims, vgg16_model_path= vgg16_model_path)
    load_model(vae, pretrained_model_file_name, model_save_folder_path)

    optim = torch.optim.Adam(vae.parameters(), lr= learning_rate, weight_decay= weight_decay)
    vae.set_optimizer(optim)

    print('Successfully built trainer class')

    trainer = Trainer(dataset_path= dataset_path, device= device)
    
    print('Successfully built trainer class')

    # Put model onto GPU if it exists
    vae.to(device)
    
    print('Starting re-training')

    train_losses = []
    val_losses = []

    # Extract epoch number from model file name
    start_epoch = int(re.search(r'E(\d+)\.pt$', pretrained_model_file_name).group(1))

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_loss = trainer.train_epoch(vae)
        val_loss = trainer.test_epoch(vae)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if print_performance_epoch_interval-1 == epoch % print_performance_epoch_interval:
            print(f'\n EPOCH {epoch + 1}/{num_epochs} \n \t train loss {train_loss} \n \t val loss {val_loss}')
            plot_ae_outputs(vae.encoder, vae.decoder, trainer.test_dataset, device, n=9)
            visualize_first_layer_filters(vae)
            visualize_deeper_layer_filter_outputs(vae, trainer.test_dataset, device, layer_index=1, num_filters_to_plot=8)
            visualize_deeper_layer_filter_outputs(vae, trainer.test_dataset, device, layer_index=2, num_filters_to_plot=8)
            visualize_deeper_layer_filter_outputs(vae, trainer.test_dataset, device, layer_index=3, num_filters_to_plot=8)
            plot_loss_progression(train_losses, val_losses, window_size=10)

        if model_save_epoch_interval-1 == epoch % model_save_epoch_interval:
            model_file_name = model_file_naming_convention(model_size, latent_dims, epoch)
            save_model(vae, model_file_name, model_save_folder_path)

if __name__ == "__main__":
    config = {
        'dataset_path': 'path_to_dataset',

        'latent_dims': 9,
        'num_epochs': 20,
        'learning_rate': 0.00005,
        'weight_decay': 1e-5,
        'model_size': 'big', 

        'model_save_folder_path': '',
        'vgg16_model_path': './models/vgg16.pth',

        'pretrained_model_file_name': 'VAE_Sbig_L9_E279.pt',  # replace with your model name

        'model_save_epoch_interval': 100,
        'print_performance_epoch_interval': 10
    }
    main(config)
