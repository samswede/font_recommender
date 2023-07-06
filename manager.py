import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
import base64
from PIL import Image
import io

from OLD_variational_autoencoder import *

class VAEModel:
    def __init__(self, model_path, embeddings_path, latent_dims):
        self.model = self.load_model(model_path, latent_dims)
        #self.df = pd.read_csv(embeddings_path)
        self.all_font_embeddings = self.load_npz(embeddings_path)

    def load_model(self, model_path, latent_dims):
        model = VariationalAutoencoder(latent_dims)  
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    
    def load_npz(self, file_path):
        with np.load(file_path) as data:
            numpy_array = data['array']

        return numpy_array

    def interpolate(self, vector_a, vector_b, fraction):
        return vector_a + fraction * (vector_b - vector_a)
    

    def generate_interpolated_images_numpy(self, font_1_index, font_2_index, interpolation_fraction):
        #font_1_vector = self.df.loc[self.df['index'] == font_1_index].values[0, 1:]
        #font_2_vector = self.df.loc[self.df['index'] == font_2_index].values[0, 1:]
        
        font_1_vector = self.all_font_embeddings[font_1_index]
        font_2_vector = self.all_font_embeddings[font_2_index]

        #font_1_vector = np.array(font_1_vector, dtype=np.float32)
        #font_2_vector = np.array(font_2_vector, dtype=np.float32)

        interpolated_vector = self.interpolate(font_1_vector, font_2_vector, interpolation_fraction)
        
        interpolated_vector_torch = torch.from_numpy(interpolated_vector).float().unsqueeze(0)
        font_1_vector_torch = torch.from_numpy(font_1_vector).float().unsqueeze(0)
        font_2_vector_torch = torch.from_numpy(font_2_vector).float().unsqueeze(0)

        interpolated_image_numpy = self.model.decoder(interpolated_vector_torch).squeeze().detach().numpy()
        font_1_image_numpy = self.model.decoder(font_1_vector_torch).squeeze().detach().numpy()
        font_2_image_numpy = self.model.decoder(font_2_vector_torch).squeeze().detach().numpy()

        return font_1_image_numpy, font_2_image_numpy, interpolated_image_numpy
    
    def generate_interpolated_images_b64(self, font_1_index, font_2_index, interpolation_fraction):
        font_1_image_numpy, font_2_image_numpy, interpolated_image_numpy = self.generate_interpolated_images_numpy(font_1_index, font_2_index, interpolation_fraction)
        font_1_image, font_2_image, interpolated_image = self.convert_numpy_arrays_to_PIL_images(font_1_image_numpy, font_2_image_numpy, interpolated_image_numpy)
        font_1_image_b64, font_2_image_b64, interpolated_image_b64 = self.convert_PIL_images_to_base64_strings(font_1_image, font_2_image, interpolated_image)

        return font_1_image_b64, font_2_image_b64, interpolated_image_b64

    def convert_numpy_arrays_to_PIL_images(self, font_1_image_numpy, font_2_image_numpy, interpolated_image_numpy):
        # Convert numpy arrays to PIL images
        font_1_image = Image.fromarray((font_1_image_numpy * 255).astype(np.uint8))
        font_2_image = Image.fromarray((font_2_image_numpy * 255).astype(np.uint8))
        interpolated_image = Image.fromarray((interpolated_image_numpy * 255).astype(np.uint8))

        return font_1_image, font_2_image, interpolated_image

    def convert_PIL_images_to_base64_strings(self, font_1_image, font_2_image, interpolated_image):
        # Convert images to base64 strings
        font_1_image_b64 = self._to_base64(font_1_image)
        font_2_image_b64 = self._to_base64(font_2_image)
        interpolated_image_b64 = self._to_base64(interpolated_image)
        
        return font_1_image_b64, font_2_image_b64, interpolated_image_b64

        
    def _to_base64(self, pil_image):
        # Helper method to convert a PIL image to a base64 string
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return "data:image/png;base64," + img_str

    def display_images(self, font1_image, font2_image, interpolated_image):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(font1_image, cmap='gray')
        axs[0].set_title('Font 1')
        axs[0].axis('off')
        axs[1].imshow(interpolated_image, cmap='gray')
        axs[1].set_title('Interpolated Font')
        axs[1].axis('off')
        axs[2].imshow(font2_image, cmap='gray')
        axs[2].set_title('Font 2')
        axs[2].axis('off')
        plt.show()

    def create_interpolation_gif(self, font_1_index, font_2_index, gif_path='interpolation.gif'):
        images = []
        for percentage in np.linspace(0, 1, num=1000):
            #font_1_vector = self.df.loc[self.df['index'] == font_1_index].values[0, 1:]
            #font_2_vector = self.df.loc[self.df['index'] == font_2_index].values[0, 1:]
            
            #font_1_vector = np.array(font_1_vector, dtype=np.float32)
            #font_2_vector = np.array(font_2_vector, dtype=np.float32)

            font_1_vector = self.all_font_embeddings[font_1_index]
            font_2_vector = self.all_font_embeddings[font_2_index]

            interpolated_vector = self.interpolate(font_1_vector, font_2_vector, percentage)
            
            interpolated_vector_torch = torch.from_numpy(interpolated_vector).float().unsqueeze(0)
            interpolated_image = self.model.decoder(interpolated_vector_torch).squeeze().detach().numpy()
            
            images.append((interpolated_image * 255).astype(np.uint8))

        imageio.mimsave(gif_path, images, duration=50)






