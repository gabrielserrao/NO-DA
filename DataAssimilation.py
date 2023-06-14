# %%

import pandas as pd
import re
import numpy as np
import torch.nn.functional as F
from timeit import default_timer

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

results_folder = '/scratch/smrserraoseabr/Projects/NO-DA/runs/TESTES'
#data assimilation parameters

x= 28
y= 3
reference_model = 193
prior_model =27 
# Define the number of optimization steps
num_steps = 100
regularization_weight = 0.0

UNKNOWN_PARAMETERS = 1 # 1 - PERMEABILITY, 2 - POROSITY

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
true = y_normalizer.decode(test_u).to(device)


# %%
#Load model and print summary
model = torch.load(path_model)
model.eval();
model.to(device) 

print(model)
#extract main parameters of model and print them like padding, number of layers, etc
print('Model loaded')
#print number of parameters of model
print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

#%%
pred = model(test_a).to(device)
pred_un = y_normalizer.decode(pred).to(device)

# %%
#list to store the predicted values and intermediate permeability values
predicted_values = []
parameter_values = []
loss_values = []
#Define inputs for optimization 
#reference data
observed = true[reference_model,:, x, y, 0]
true_map = a_normalizer.decode(test_a)[reference_model, -1,:, :, UNKNOWN_PARAMETERS]


#prior data
prior_model_inputs = test_a[prior_model,:,:, :, :] 
prior_model_inputs = prior_model_inputs.unsqueeze(0)

predicted =  pred_un.detach().numpy()[prior_model,:, x, y, 0]
initial_map = a_normalizer.decode(test_a)[prior_model, -1,:, :, UNKNOWN_PARAMETERS]


fig, main_ax = plt.subplots()

# Plot all models in light gray

time =pred_un[0,:,0,0,0].detach().numpy()



main_ax.plot(time, true.detach().numpy() [reference_model,:, x, y, 0], color='red', label = 'Reference case - true')
main_ax.plot(time, pred_un.detach().numpy() [reference_model,:, x, y, 0], color='red', linestyle ='--',  label = 'Reference case - FNO') 
main_ax.plot(time, true.detach().numpy() [prior_model,:, x, y, 0], color='blue', label = 'Prior case - true')
main_ax.plot(time, pred_un.detach().numpy() [prior_model,:, x, y, 0], color='blue', linestyle ='--',  label = 'Prior case - FNO')
main_ax.legend()


main_ax.set_xlabel('Time')
main_ax.legend()

# Set labels and title for the main figure
main_ax.set_xlabel('Time')
main_ax.set_ylabel(variable)
main_ax.set_title(f'Montintoring {variable} at x= {x} and y = {y} ')

# Create inset of width 30% and height 30% at the upper right corner of the main plot
left, bottom, width, height = [0.2, 0.4, 0.2, 0.2] # adjust as needed
inset_ax = fig.add_axes([left, bottom, width, height])
im = inset_ax.imshow(test_a[reference_model,-1, :, :, 0], cmap='jet')

inset_ax.scatter(x, y, s=20, edgecolor='red', facecolor='none', linewidth=2)

#fix scale of main plot between 0 and 1 
inset_ax.axis('off')

#save fig on results folder
plt.savefig(os.path.join(results_folder, f'Initial_overview_prior_{prior_model}_reference_{reference_model}_x{x}_y{y}.png'))

plt.show()
plt.close()



# %%
from torch import optim
import pickle
import os
#%%
# Define the loss function as the difference between observed data and predicted data

prior_response = pred_un.detach().numpy()[prior_model,:, x, y, 0]
reference_response = true.detach().numpy() [reference_model,:, x, y, 0]
# Initialize the optimizer, we will use Adam here
optimizer = optim.Adam([prior_model_inputs[:,:,:,UNKNOWN_PARAMETERS]], lr=0.01)  # Adjust learning rate as needed


fig, ax = plt.subplots()

for step in range(num_steps):
    print(f'Step {step} of {num_steps}')
    t1 = default_timer()
    optimizer.zero_grad()  # Clear previous gradients
    pred = model(prior_model_inputs)
 
    pred_un = y_normalizer.decode(pred)[0,:,x,y,0]   

    if regularization_weight > 0.0:
        loss = F.mse_loss(observed, pred_un, reduction='mean') + regularization_weight * torch.norm(prior_model_inputs[:,:,:,UNKNOWN_PARAMETERS])**2
    else:    
        loss = F.mse_loss(observed, pred_un, reduction='mean')

    # Compute the loss WITH A SINGLE POINT
    #loss = F.mse_loss(observed, pred_un, reduction='mean') + regularization_weight * torch.norm(prior_model_inputs[:,:,:,1])**2
    

    loss.backward()  # Compute the gradients
    optimizer.step()  # Update the parameters using the gradients      

    # Store the current predicted values
    predicted_values.append(pred_un.detach().numpy())

    # Store the current permeability values
    parameter_values.append(prior_model_inputs[0, -1,:, :, UNKNOWN_PARAMETERS].detach().numpy())

    #apend loss to list
    loss_values.append(loss.item())
    t2 = default_timer()

        #print mse, l2 for train and test data for each epoch
    print(f'ep {step}: t={t2-t1:.3f}, train_mse={loss.item():.3e}')
    #save loss on disk
    with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_loss_values_step.txt'), 'w') as f:
        f.write(f'epoch {step}: t={t2-t1:.3f}, train_mse={loss.item():.3e}\n')

       
    if step % 10 == 0:
        fig, main_ax = plt.subplots()
        # Plot all models in light gray   

        main_ax.plot(time, reference_response, color='red', label = 'Reference case - true')
        main_ax.plot(time, prior_response, color='blue', linestyle ='--',  label = 'Prior case - FNO') 
        #include posterior case only after 10 steps
        main_ax.plot(time, y_normalizer.decode(pred).detach().numpy()[0,:, x, y, 0], color='green', linestyle ='--',  label = 'Posterior case - FNO')
        main_ax.legend()


        main_ax.set_xlabel('Time')
        main_ax.legend()

        # Set labels and title for the main figure
        main_ax.set_xlabel('Time')
        main_ax.set_ylabel(variable)
        main_ax.set_title(f'Montintoring {variable} at x= {x} and y = {y} ')

       #save fig on results folder
        plt.savefig(os.path.join(results_folder, f'Initial_overview_prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_step{step}.png'))

        plt.show()
        plt.savefig(os.path.join(image_folder, f'Optimim_{step}.png'))
        plt.close()

        #plot current  parameter_values
        fig, ax = plt.subplots()
        ax.imshow(prior_model_inputs[0, -1,:, :, UNKNOWN_PARAMETERS].detach().numpy(), cmap='jet')
        ax.set_title(f'Permeability values at step {step}')
        plt.show()
        plt.savefig(os.path.join(results_folder, f'Permeability_{step}.png'))       
        plt.close()

        #plot current  observed and predicted values pred_un

        fig, main_ax = plt.subplots()
        # Plot all models in light gray   

        main_ax.plot(time, observed, color='red', label = 'Reference case - true')
        main_ax.plot(time, pred_un, color='blue', linestyle ='--',  label = 'Prior case - FNO') 
        #include posterior case only after 10 steps
        main_ax.plot(time, y_normalizer.decode(pred).detach().numpy()[0,:, x, y, 0], color='green', linestyle ='--',  label = 'Posterior case - FNO')
        main_ax.legend()
        main_ax.set_xlabel('Time')
        main_ax.legend()
        # Set labels and title for the main figure
        main_ax.set_xlabel('Time')
        main_ax.set_ylabel(variable)
        main_ax.set_title(f'Montintoring {variable} at x= {x} and y = {y} ')

       #save fig on results folder
        plt.savefig(os.path.join(results_folder, f'Initial_overview_prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_step{step}_used_loss.png'))

        plt.show()
        plt.savefig(os.path.join(image_folder, f'Optimim_{step}.png'))
        plt.close()


        #save predicted_values , permeability_values and loss on disk as pikle files
        with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_predicted_values_step{step}.pkl'), 'wb') as f:
            pickle.dump(predicted_values, f)
        with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_permeability_values_step{step}.pkl'), 'wb') as f:
            pickle.dump(parameter_values, f)
        with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_loss_values_step{step}.pkl'), 'wb') as f:
            pickle.dump(loss_values, f)

#save predicted_values , permeability_values and loss on disk as pikle files
with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_predicted_values_step{step}.pkl'), 'wb') as f:
    pickle.dump(predicted_values, f)
with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_permeability_values_step{step}.pkl'), 'wb') as f:
    pickle.dump(parameter_values, f)
with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_loss_values_step{step}.pkl'), 'wb') as f:
    pickle.dump(loss_values, f)



#      






# %%
