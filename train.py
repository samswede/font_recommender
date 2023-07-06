import torch
from utils import *
from preprocessing import *
from NEW_variational_autoencoder import *

""" TO DO:
   
      - make save_model and load_model into methods in the vae class
      - make train_model a method within vae class, too. Then you can put device and everything in there.


"""

# Define the training loop
def train_model(vae, device, train_loader, valid_loader, test_dataset, optimizer, num_epochs):
    for epoch in range(num_epochs):
        train_loss = train_epoch(vae, device, train_loader, optimizer)
        val_loss = test_epoch(vae, device, valid_loader)
        if 9 == epoch % 10:
            print(f'\n EPOCH {epoch + 1}/{num_epochs} \n \t train loss {train_loss} \n \t val loss {val_loss}')
            plot_ae_outputs(vae.encoder, vae.decoder, test_dataset, device, n=9)
        if 99 == epoch % 100:
            save_model(vae, config['model_size'], config['latent_dims'], epoch)

# Define the model saving function
def save_model(model, model_size, d, epoch):
    model_save_name = '{}_vae_L{}_E{}.pt'.format(model_size, d, epoch+1)
    model_path = F"/content/drive/MyDrive/FontFinder/Models/VAEs/{model_save_name}" 
    torch.save(model.state_dict(), model_path)

# Define the main function
def main(config):
    train_loader, valid_loader, test_loader, train_dataset, test_dataset, transform_norm = prepare_data_loaders(config['dataset_path'])

    torch.manual_seed(0)
    vae = VariationalAutoencoder(latent_dims=config['latent_dims'])
    optim = torch.optim.Adam(vae.parameters(), lr=config['learning_rate'], weight_decay=1e-5)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')
    vae.to(device)
    
    train_model(vae, device, train_loader, valid_loader, test_dataset, optim, config['num_epochs'])

if __name__ == "__main__":
    config = {
        'dataset_path': 'path_to_dataset',
        'latent_dims': 9,
        'num_epochs': 20,
        'learning_rate': 0.00005,
        'model_size': 'big'
    }
    main(config)
