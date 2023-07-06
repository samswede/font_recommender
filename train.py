import torch
from utils import *
from NEW_variational_autoencoder import *
from trainer import Trainer

""" TO DO:
   
      - make save_model and load_model into methods in the vae class
      - make train_model a method within vae class, too. Then you can put device and everything in there.


"""


# Define the naming convention
def model_file_naming_convention(model_size, latent_dims, epoch):
    model_file_name = f'VAE_S{model_size}_L{latent_dims}_E{epoch}.pt'
    return model_file_name

# Define the model saving function
def save_model(model, model_file_name, model_save_folder_path):
    model_path = F'{model_save_folder_path}/{model_file_name}'
    torch.save(model.state_dict(), model_path)


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

    print_performance_epoch_interval = config['print_performance_epoch_interval']
    model_save_epoch_interval = config['model_save_epoch_interval']

    # Set replicable random seed
    torch.manual_seed(0)

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Instantiate classes
    vae = VariationalAutoencoder(latent_dims= latent_dims, vgg16_model_path= vgg16_model_path)
    optim = torch.optim.Adam(vae.parameters(), lr= learning_rate, weight_decay= weight_decay)
    vae.set_optimizer(optim)

    print('Successfully built trainer class')

    trainer = Trainer(dataset_path= dataset_path, device= device)
    
    print('Successfully built trainer class')

    # Put model onto GPU if it exists
    vae.to(device)
    
    print('Starting training')
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(vae)
        val_loss = trainer.test_epoch(vae)
        if print_performance_epoch_interval-1 == epoch % print_performance_epoch_interval:
            print(f'\n EPOCH {epoch + 1}/{num_epochs} \n \t train loss {train_loss} \n \t val loss {val_loss}')
            plot_ae_outputs(vae.encoder, vae.decoder, trainer.test_dataset, device, n=9)
            visualize_first_layer_filters(vae)
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

        'model_save_epoch_interval': 100,
        'print_performance_epoch_interval': 10
    }
    main(config)
