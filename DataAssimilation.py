# %%

import pandas as pd
import re
import numpy as np
import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from IPython.display import Image as DisplayImage


from utilities import *
from model_fourier_3d import *



# %%


# DATASET
data_folder = "/scratch/smrserraoseabr/Projects/FluvialCO2/results32/"
num_files = 1000
traintest_split = 0.8

#CASE CONFIGURATION
input_vars = ['Por', 'Perm', 'gas_rate'] # Porosity, Permeability, ,  Well 'gas_rate', Pressure + x, y, time encodings 
variable = 'Pressure'
output_vars = [variable]
case_name = 'VINI_ns_fourier_3d_N800.0_ep500_m12_w128_b61_padding4' + '_' + variable
#DEVICE SETTINGS
device = 'cpu'
#OUTPUT CONFIGURATION
plot_model_eval =True
plot_comparison = True
plot_lines = True
plot_gifs =True

#DATA ASSIMILATION CONFIGURATION

results_folder = '/scratch/smrserraoseabr/Projects/NO-DA/runs/TESTES/images'
#data assimilation parameters
x= 28
y= 3
reference_model = 193
prior_model =27 
# Define the number of optimization steps
num_steps = 1000
###############################################



path_runs = os.path.join('runs', case_name)
path_model = os.path.join(path_runs, 'model', f'{case_name}_model.pt')
path_normalizer = os.path.join(path_runs, 'model')
image_folder = os.path.join(path_runs, 'images')
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

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


if variable == 'CO_2':
    colorbar_vmax, colorbar_vmin = 1.0, 0.0 # Define your min and max here
elif variable == 'Pressure':
    colorbar_vmin, colorbar_vmax = 200.0, 600.0 # Define your min and max here
  # Change this to the index you want
      


# %%
#LOAD NORMALIZATION PARAMETERS
dataset = ReadXarray(folder=data_folder, input_vars=input_vars, output_vars=output_vars, num_files = num_files, traintest_split = traintest_split)

# Get input and output data tensors
train_a = dataset.train_data_input
train_u = dataset.train_data_output

test_a = dataset.test_data_input
test_u = dataset.test_data_output

print(f'Input train data shape: {train_a.shape}')
print(f'Output train data shape: {train_u.shape}')

print(f'Input test data shape: {test_a.shape}')
print(f'Output test data shape: {test_u.shape}')


# Move data tensors to GPU if available
train_a = train_a.to(device)
train_u = train_u.to(device)

test_a = test_a.to(device)
test_u = test_u.to(device)

# Normalize input_data and output_data
#check if normalizer exists and load it, otherwise raise a warning and create a new one
if os.path.exists(os.path.join((path_normalizer), f'{case_name}_a_normalizer_mean.pt')):
    a_normalizer_mean = torch.load(os.path.join((path_normalizer), f'{case_name}_a_normalizer_mean.pt'))
    a_normalizer_std = torch.load(os.path.join((path_normalizer), f'{case_name}_a_normalizer_std.pt'))
    y_normalizer_mean = torch.load(os.path.join((path_normalizer), f'{case_name}_y_normalizer_mean.pt'))
    y_normalizer_std = torch.load(os.path.join((path_normalizer), f'{case_name}_y_normalizer_std.pt'))
    print('Normalizer loaded')
    a_normalizer = UnitGaussianNormalizer(test_a, mean=a_normalizer_mean, std=a_normalizer_std)
    y_normalizer = UnitGaussianNormalizer(test_u, mean=y_normalizer_mean, std=y_normalizer_std)


else:
    print('Normalizer not found, creating a new one and saving it')
    a_normalizer = UnitGaussianNormalizer(train_a)
    y_normalizer = UnitGaussianNormalizer(train_u)
    torch.save(a_normalizer.mean, os.path.join((path_normalizer), f'{case_name}_a_normalizer_mean.pt'))
    torch.save(a_normalizer.std, os.path.join((path_normalizer), f'{case_name}_a_normalizer_std.pt'))

    torch.save(y_normalizer.mean, os.path.join((path_normalizer), f'{case_name}_y_normalizer_mean.pt'))
    torch.save(y_normalizer.std, os.path.join((path_normalizer), f'{case_name}_y_normalizer_std.pt'))


train_a= a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

train_u = y_normalizer.encode(train_u)
test_u = y_normalizer.encode(test_u)

#DEFINE TRUE DATA - RESULTS OF DARTS SIMULATION
true = y_normalizer.decode(test_u)


# %%
#Load model and print summary
model = torch.load(path_model)
model.eval(); 
print(model)
#extract main parameters of model and print them like padding, number of layers, etc
print('Model loaded')
#print number of parameters of model
print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')


pred = model(test_a)
pred_un = y_normalizer.decode(pred)






# %%
#list to store the predicted values and intermediate permeability values
predicted_values = []
permeability_values = []
loss_values = []
#Define inputs for optimization 
#reference data
observed = true[reference_model,:, x, y, 0]
true_perm_map = a_normalizer.decode(test_a)[reference_model, -1,:, :, 1]


#prior data
prior_model_inputs = test_a[prior_model,:,:, :, :] 
prior_model_inputs = prior_model_inputs.unsqueeze(0)

predicted =  pred_un.detach().numpy()[prior_model,:, x, y, 0]
initial_perm_map = a_normalizer.decode(test_a)[prior_model, -1,:, :, 1]

# %%
from torch import optim
import pickle
import os
#%%
# Define the loss function as the difference between observed data and predicted data



#regularization_weight = 0.01


# Initialize the optimizer, we will use Adam here
optimizer = optim.Adam([prior_model_inputs[:,:,:,1]], lr=0.01)  # Adjust learning rate as needed




DA_mse= 0.0

fig, ax = plt.subplots()

for step in range(num_steps):
    print(f'Step {step} of {num_steps}')
    optimizer.zero_grad()  # Clear previous gradients
    pred = model(prior_model_inputs)
    pred_un = y_normalizer.decode(pred)[0,:,x,y,0]   
    
    loss = F.mse_loss(observed, pred_un, reduction='mean')

    # Compute the loss WITH A SINGLE POINT
    #loss = loss_fn(observed, pred_un) + regularization_weight * torch.norm(prior_model_inputs[:,:,:,1])**2
    

    loss.backward()  # Compute the gradients
    optimizer.step()  # Update the parameters using the gradients


      

    # Store the current predicted values
    predicted_values.append(pred_un.detach().numpy())

    # Store the current permeability values
    permeability_values.append(prior_model_inputs[0, -1,:, :, 1].detach().numpy())

    #apend loss to list
    loss_values.append(loss.item())

       
    if step % 10 == 0:
        ax.clear()
        ax.plot(observed.detach().numpy(), label='Reference')
        ax.plot(pred_un.detach().numpy(), label='Posterior')
        ax.legend()
        #save image
        plt.savefig(os.path.join(image_folder, f'Optimim_{step}.png'))

        #save predicted_values , permeability_values and loss on disk as pikle files
        with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_predicted_values_step{step}.pkl'), 'wb') as f:
            pickle.dump(predicted_values, f)
        with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_permeability_values_step{step}.pkl'), 'wb') as f:
            pickle.dump(permeability_values, f)
        with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_loss_values_step{step}.pkl'), 'wb') as f:
            pickle.dump(loss_values, f)

#save predicted_values , permeability_values and loss on disk as pikle files
with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_predicted_values_step{step}.pkl'), 'wb') as f:
    pickle.dump(predicted_values, f)
with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_permeability_values_step{step}.pkl'), 'wb') as f:
    pickle.dump(permeability_values, f)
with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_loss_values_step{step}.pkl'), 'wb') as f:
    pickle.dump(loss_values, f)



#      





