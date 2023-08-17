# %%
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
import pandas as pd
import re
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from utilities import *
import imageio
from io import BytesIO
from IPython.display import Image as DisplayImage
from model_fourier_3d import *
import torchmetrics
print(torch.__version__)
print(f"GPUs:{torch.cuda.device_count()}")
import os
print(os.getcwd())

# %%
###############################################
#INTIAL CONFIGS
# DATASET

TAG = 'MonthQgWellCenter'
FOLDER = '/samoa/data/smrserraoseabr/NO-DA/dataset/DARTS/runnedmodels/filtered' #'/samoa/data/smrserraoseabr/NO-DA/dataset/mixedcontext32x32' #"../dataset/DARTS/runnedmodels_wells/filtered"  #"/nethome/atena_projetos/bgy3/NO-DA/datasets/results" + str(resolution) + "/"
INPUT_VARS = ['Por', 'Perm', 'gas_rate'] # Porosity, Permeability, ,  Well 'gas_rate', Pressure + x, y, time encodings 
OUTPUT_VARS = ['Pressure'] 
#CONFIGS OF THE MODEL TO GENERATE RESULTS
BASE_PATH = '/samoa/data/smrserraoseabr/NO-DA/runs'
NUM_FILES= 1000
TRAINTEST_SPLIT = 0.8
BATCH_SIZE = 1
EPOCHS = 100
MODES = 18
WIDTH = 128
NORMALIZER = 'PointGaussianNormalizer'
WELLS_POSITIONS = True
#List of samples to plot:
BATCH_TO_PLOT = [0, 1, 2, 3, 4, 5, 6, 7, 8]
SAMPLES_TO_PLOT = [0]#, 1, 2, 3, 4, 5, 6, 7, 8]
#DEVICE SETTINGS
device = 'cpu'
#OUTPUT CONFIGURATION
EVALUATE_METRICS = False
plot_model_eval = True
plot_comparison = True
plot_lines =True
plot_gifs =True
###############################################
variable = OUTPUT_VARS[0]
ntrain = NUM_FILES * TRAINTEST_SPLIT
ntest = NUM_FILES - ntrain

path = 'FNO_3d_{}_N{}_ep{}_m{}_w{}_b{}_norm{}'.format(TAG,ntrain, EPOCHS, MODES, WIDTH, BATCH_SIZE, NORMALIZER)
path += '_INPUT_' + '_'.join(INPUT_VARS) + '_OUTPUT_' + '_'.join(OUTPUT_VARS)
#path = '/samoa/data/smrserraoseabr/NO-DA/runs/FNO_3d_N800.0_ep200_m18_w128_b10_INPUT_Por_Perm_gas_rate_OUTPUT_CO_2'
path_runs = os.path.join(BASE_PATH, path)
path_model = os.path.join(path_runs, f'{path}_model.pt')
#path_model='/samoa/data/smrserraoseabr/NO-DA/runs/FNO_3d_N800.0_ep200_m18_w128_b10_INPUT_Por_Perm_gas_rate_OUTPUT_CO_2/FNO_3d_N800.0_ep200_m18_w128_b10_INPUT_Por_Perm_gas_rate_OUTPUT_CO_2_model.pt'
path_normalizer = path_runs
image_folder = os.path.join(path_runs, 'images')
log_folder = os.path.join(path_runs, 'log')

#check  all paths
print(f'Path runs: {path_runs}')
print(f'Path model: {path_model}')

#CHCEK IF FOLDERS EXISTS
if not os.path.exists(path_runs):
    raise ValueError(f'Path {path_runs} does not exist.')
if not os.path.exists(path_model):
       raise ValueError(f'Path {path_model} does not exist.')
if not os.path.exists(image_folder):
    raise ValueError(f'Path {image_folder} does not exist.')

if not os.path.exists(image_folder):
    os.makedirs(image_folder)


if variable == 'CO_2':
    colorbar_vmax, colorbar_vmin = 1.0, 0.0 # Define your min and max here
elif variable == 'Pressure':
    colorbar_vmin, colorbar_vmax = 200.0, 300.0 # Define your min and max here
  # Change this to the index you want
#%%     
###############################################
#LOAD DATA

# Create instance of ReadXarrayDatasetNorm class for training data
dataset = ReadXarrayDataset(folder=FOLDER, input_vars=INPUT_VARS, output_vars=OUTPUT_VARS, num_files = NUM_FILES, wells_positions=WELLS_POSITIONS)

train_size = int(TRAINTEST_SPLIT * len(dataset))
test_size = len(dataset) - train_size


train_loader = DataLoader(torch.utils.data.Subset(dataset, range(0, train_size)),
                           batch_size=BATCH_SIZE,
                             shuffle=False)
test_loader = DataLoader(torch.utils.data.Subset(dataset, range(train_size, train_size + test_size)), 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)
# We no longer have the entire dataset loaded into memory. The normalization is handled by the Dataset class.

# %%
#LOAD NORMALIZATION PARAMETERS

# Normalize input_data and output_data
#check if normalizer exists and load it, otherwise raise a warning and create a new one
if os.path.exists(os.path.join(path_runs,'normalizer_mean_input.pt')):
    input_normalizer_mean = torch.load(os.path.join(path_runs,'normalizer_mean_input.pt'))
    input_normalizer_std = torch.load(os.path.join(path_runs,'normalizer_std_input.pt'))
    output_normalizer_mean = torch.load(os.path.join(path_runs,'normalizer_mean_output.pt'))
    output_normalizer_std = torch.load(os.path.join(path_runs,'normalizer_std_output.pt'))
    print('Normalizer loaded')
    #create a new normalizer
    if NORMALIZER == 'PointGaussianNormalizerNoNaN':
        input_normalizer = PointGaussianNormalizerNoNaN(train_loader, mean = input_normalizer_mean, std = input_normalizer_std, is_label=False)
        output_normalizer = PointGaussianNormalizerNoNaN(train_loader, mean = output_normalizer_mean, std = output_normalizer_std, is_label=True)
    elif NORMALIZER == 'PointGaussianNormalizer':
        input_normalizer = PointGaussianNormalizer(train_loader, mean = input_normalizer_mean, std = input_normalizer_std, is_label=False)
        output_normalizer = PointGaussianNormalizer(train_loader, mean = output_normalizer_mean, std = output_normalizer_std, is_label=True)

    input_normalizer = input_normalizer.cuda(device)
    output_normalizer = output_normalizer.cuda(device)

else:
    print('Normalizer not found')
    #raise 
    ValueError(f'Normalizer not found in {path_runs}')

#%%
###############################################
device = torch.device("cpu")
model = torch.load(path_model, map_location=device).to(device)  # load the model to CPU
model.eval()  # set the model to evaluation mode

for batch_idx, (x, y) in enumerate(test_loader):        

            #x is the input data, (batch_size, 61,32,32,6), where dim 1 is permeability 
            x = x.to(device)
            true_y = y.to(device)
            x = input_normalizer.encode(x)
            out = model(x)
            out = output_normalizer.decode(out).detach().cpu() #pressure (batch_size, 61,32,32,1) the last dimension is the Pressure 

            

