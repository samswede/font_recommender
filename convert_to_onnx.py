import torch

from NEW_variational_autoencoder import VariationalAutoencoder

def convert():
    vae_model = VariationalAutoencoder(latent_dims=12, vgg16_model_path='./models/vgg16.pth')
    vae_model.load_state_dict(torch.load('./models/VAE_Sbatch50_beta0001_L12_E949.pt', map_location=torch.device('cpu')))
    encoder = vae_model.encoder
    encoder.eval()

    dummy_input = torch.randn(1, 1, 128, 128)
    torch.onnx.export(encoder, dummy_input, "./models/encoder_L12.onnx", verbose=True)
    pass

if __name__ == '__main__':
    convert()