
# Import webapp libraries
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

# Import personalised modules
from vector_database import *
from utils import *
from manager import *
from OLD_variational_autoencoder import *

# Import module to retrieve data from google drive
import gdown

# Download the model from Google Drive
#url = 'https://drive.google.com/uc?id=<your-file-id>'
#output = 'vae_model.pt'
#gdown.download(url, output, quiet=False)

#===================================================================
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#===================================================================

# Initialize the FastAPI application
app = FastAPI()

# Serve static files from the "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware setup to allow requests from the specified origins
origins = [
    "http://localhost",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Jinja2 templates with the "templates" directory
templates = Jinja2Templates(directory="templates")


#====================================================================================================================
# Load Model
#====================================================================================================================

model_path='./models/big_vae_L9_E700.pt'
embeddings_path='./data/embeddings/cleaned_big_L9_E700.csv'
font_embeddings_path= './data/embeddings/all_font_embeddings.npz'

vae = VAEModel(model_path= model_path, 
               embeddings_path= font_embeddings_path, 
               latent_dims=9)

#====================================================================================================================
# Load Embeddings and make Vector Database
#====================================================================================================================

"""
# load font embeddings as pandas df
df = pd.read_csv(embeddings_path)
# drop index column
df = df.drop(df.columns[0], axis=1)
# Convert the DataFrame to a numpy array
font_embeddings_array = df.values
#dict_font_labels_to_indices = load_data_dict(f'{data_path}dict_font_labels_to_indices')
dict_font_labels_to_indices = {i: i for i in range(font_embeddings_array.shape[0])}
dict_font_indices_to_labels = {v: k for k, v in dict_font_labels_to_indices.items()}
"""


font_embeddings_path = './data/embeddings/all_font_embeddings.npz'
font_embeddings_array = load_npz(file_path= font_embeddings_path)

dictionary_path = './data/embeddings/font_name_to_index.pickle'
dict_font_labels_to_indices= load_data_dict(dictionary_path)
dict_font_indices_to_labels = {v: k for k, v in dict_font_labels_to_indices.items()}


all_font_labels = [label for label, index in dict_font_labels_to_indices.items()]


metrics = ['angular', 'euclidean', 'manhattan', 'hamming', 'dot']

font_vector_db = MultiMetricDatabase(dimensions=font_embeddings_array.shape[1], metrics= metrics, n_trees=30)

# Add all fonts to vector database
font_vector_db.add_vectors(font_embeddings_array, dict_font_labels_to_indices)


#====================================================================================================================
# Define core recommendation function
#====================================================================================================================

def find_similar_fonts(chosen_font_label, distance_metric='euclidean'):
    
    # Translate indication name to index in indication diffusion profiles, to retrieve diffusion profile
    #chosen_font_label = graph_manager.mapping_indication_name_to_label[chosen_indication_name]
    chosen_font_index = dict_font_labels_to_indices[chosen_font_label]
    chosen_font_embedding = font_embeddings_array[chosen_font_index]

    #====================================
    # Querying Vector Database to return drug candidates
    #====================================
    num_recommendations = 10

    query = chosen_font_embedding

    font_candidates_indices = font_vector_db.nearest_neighbors(query, distance_metric, num_recommendations)

    font_candidates_labels = [dict_font_indices_to_labels[index] for index in font_candidates_indices]
    #drug_candidates_names = [graph_manager.mapping_drug_label_to_name[i] for i in font_candidates_labels]

    return font_candidates_labels # List


#===================================================================
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#===================================================================

# Define a Pydantic model for font, similar fonts request, and interpolation request
class Font(BaseModel):
    value: int
    name: str

class SimilarFontsRequest(BaseModel):
    font_index: int

class InterpolationRequest(BaseModel):
    font_1_index: int
    font_2_index: int
    interpolation_fraction: float

#====================================================================================================================
# Define application routes
#====================================================================================================================

@app.get("/", response_class=HTMLResponse)
async def read_items(request: Request):
    """Serve the index.html page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/fonts", response_model= List[Font])
async def get_fonts():
    """Return a list of fonts"""
    list_of_fonts = [
        {"value": dict_font_labels_to_indices[label], "name": label}
        for label in all_font_labels
    ]
    return list_of_fonts

@app.post("/similar_fonts", response_model= List[Font])
async def get_similar_fonts(similar_fonts_request: SimilarFontsRequest):
    """Return a list of drugs based on the selected disease"""

    assert type(similar_fonts_request.font_index) == int

    chosen_font_label = dict_font_indices_to_labels[similar_fonts_request.font_index]

    font_candidates = find_similar_fonts(chosen_font_label=chosen_font_label, distance_metric='euclidean')
    list_of_font_candidates = [
        {"value": dict_font_labels_to_indices[label], "name": label}
        for label in font_candidates
    ]
    return list_of_font_candidates

@app.post("/interpolation", response_class=JSONResponse)
async def get_interpolation_data(request: InterpolationRequest):
    # Extract parameters from request
    #font_1_label = request.font_1_label
    #font_2_label = request.font_2_label

    font_1_index = request.font_1_index
    font_2_index = request.font_2_index

    interpolation_fraction = request.interpolation_fraction

    #font_1_index = dict_font_labels_to_indices[font_1_label]
    #font_2_index = dict_font_labels_to_indices[font_2_label]

    # Generate interpolated images
    font_1_image_b64, font_2_image_b64, interpolated_image_b64 = vae.generate_interpolated_images_b64(font_1_index, font_2_index, interpolation_fraction)
    
    # Create the response
    response = {
        "font_1_image": font_1_image_b64,
        "interpolated_image": interpolated_image_b64,
        "font_2_image": font_2_image_b64
    }

    return response
