#%%
import pandas as pd
import re
import numpy as np
import torch.nn.functional as F
from torch import optim
import pickle
from timeit import default_timer
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from IPython.display import Image as DisplayImage
from utilities import *
from model_fourier_3d import *

# DATASET
data_folder = "/scratch/smrserraoseabr/Projects/FluvialCO2/results32/"
num_files = 1000
traintest_split = 0.8

# CASE CONFIGURATION
input_vars = ['Por', 'Perm', 'gas_rate']  # Porosity, Permeability, Well 'gas_rate', Pressure + x, y, time encodings
variable = 'Pressure'
output_vars = [variable]
case_name = 'VINI_ns_fourier_3d_N800.0_ep500_m12_w128_b61_padding4' + '_' + variable

# DEVICE SETTINGS
device = 'cpu'

# OUTPUT CONFIGURATION
plot_model_eval = True
plot_comparison = True
plot_lines = True
plot_gifs = True

# DATA ASSIMILATION CONFIGURATION
results_folder = '/scratch/smrserraoseabr/Projects/NO-DA/runs/TESTES'
# data assimilation parameters
x = 28
y = 3
reference_model = 193
prior_model = 27
# Define the number of optimization steps
num_steps = 100
regularization_weight = 0.0
learning_rate = 0.01  # ADAM learning rate
UNKNOWN_PARAMETERS = 1  # 1 - PERMEABILITY, 2 - POROSITY

path_runs = os.path.join('runs', case_name)
path_model = os.path.join(path_runs, 'model', f'{case_name}_model.pt')
path_normalizer = os.path.join(path_runs, 'model')
image_folder = os.path.join(path_runs, 'images')

# Check if all paths exist
print(f'Path runs: {path_runs}')
print(f'Path model: {path_model}')

# Check if folders exist
if not os.path.exists(path_runs):
    raise ValueError(f'Path {path_runs} does not exist.')
if not os.path.exists(path_model):
    raise ValueError(f'Path {path_model} does not exist.')
if not os.path.exists(image_folder):
    raise ValueError(f'Path {image_folder} does not exist.')

if variable == 'CO_2':
    colorbar_vmax, colorbar_vmin = 1.0, 0.0  # Define your min and max here
elif variable == 'Pressure':
    colorbar_vmin, colorbar_vmax = 200.0, 600.0  # Define your min and max here

# LOAD NORMALIZATION PARAMETERS
dataset = ReadXarray(
    folder=data_folder,
    input_vars=input_vars,
    output_vars=output_vars,
    num_files=num_files,
    traintest_split=traintest_split
)

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
# Check if normalizer exists and load it, otherwise create a new one
normalizer_file_path = os.path.join(path_normalizer, f'{case_name}_a_normalizer_mean.pt')
if os.path.exists(normalizer_file_path):
    a_normalizer_mean = torch.load(normalizer_file_path)
    a_normalizer_std = torch.load(os.path.join(path_normalizer, f'{case_name}_a_normalizer_std.pt'))
    y_normalizer_mean = torch.load(os.path.join(path_normalizer, f'{case_name}_y_normalizer_mean.pt'))
    y_normalizer_std = torch.load(os.path.join(path_normalizer, f'{case_name}_y_normalizer_std.pt'))
    print('Normalizer loaded')
else:
    print('Normalizer not found, creating a new one and saving it')
    a_normalizer = UnitGaussianNormalizer(train_a)
    y_normalizer = UnitGaussianNormalizer(train_u)
    torch.save(a_normalizer.mean, os.path.join(path_normalizer, f'{case_name}_a_normalizer_mean.pt'))
    torch.save(a_normalizer.std, os.path.join(path_normalizer, f'{case_name}_a_normalizer_std.pt'))
    torch.save(y_normalizer.mean, os.path.join(path_normalizer, f'{case_name}_y_normalizer_mean.pt'))
    torch.save(y_normalizer.std, os.path.join(path_normalizer, f'{case_name}_y_normalizer_std.pt'))

a_normalizer = UnitGaussianNormalizer(test_a, mean=a_normalizer_mean, std=a_normalizer_std)
y_normalizer = UnitGaussianNormalizer(test_u, mean=y_normalizer_mean, std=y_normalizer_std)

train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

train_u = y_normalizer.encode(train_u)
test_u = y_normalizer.encode(test_u)

# DEFINE TRUE DATA - RESULTS OF DARTS SIMULATION
true = y_normalizer.decode(test_u).to(device)
#%%
# Load model and print summary
model = torch.load(path_model)
model.eval()
model.to(device)

print(model)
# Extract main parameters of model and print them like padding, number of layers, etc
print('Model loaded')
# Print number of parameters of model
print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
#%%
pred = model(test_a).to(device)
pred_un = y_normalizer.decode(pred).to(device)

predicted_values = []
parameter_values = []
loss_values = []

observed = true[reference_model, :, x, y, 0]
true_map = a_normalizer.decode(test_a)[reference_model, -1, :, :, UNKNOWN_PARAMETERS]

prior_model_inputs = test_a[prior_model, :, :, :, :]
prior_model_inputs = prior_model_inputs.unsqueeze(0)
prior_model_inputs_leaf = torch.tensor(prior_model_inputs[:, :, :, UNKNOWN_PARAMETERS], requires_grad=True)



predicted = pred_un.detach().numpy()[prior_model, :, x, y, 0]
initial_map = a_normalizer.decode(test_a)[prior_model, -1, :, :, UNKNOWN_PARAMETERS]

fig, main_ax = plt.subplots()

time = pred_un[0, :, 0, 0, 0].detach().numpy()

main_ax.plot(time, true.detach().numpy()[reference_model, :, x, y, 0], color='red', label='Reference case - true')
main_ax.plot(time, pred_un.detach().numpy()[reference_model, :, x, y, 0], color='red', linestyle='--', label='Reference case - FNO')
main_ax.plot(time, true.detach().numpy()[prior_model, :, x, y, 0], color='blue', label='Prior case - true')
main_ax.plot(time, pred_un.detach().numpy()[prior_model, :, x, y, 0], color='blue', linestyle='--', label='Prior case - FNO')
main_ax.legend()

main_ax.set_xlabel('Time')
main_ax.legend()
main_ax.set_xlabel('Time')
main_ax.set_ylabel(variable)
main_ax.set_title(f'Montintoring {variable} at x={x} and y={y}')

left, bottom, width, height = [0.2, 0.4, 0.2, 0.2]  # Adjust as needed
inset_ax = fig.add_axes([left, bottom, width, height])
im = inset_ax.imshow(test_a[reference_model, -1, :, :, 0], cmap='jet')
inset_ax.scatter(x, y, s=20, edgecolor='red', facecolor='none', linewidth=2)
inset_ax.axis('off')

plt.savefig(os.path.join(results_folder, f'Initial_overview_prior_{prior_model}_reference_{reference_model}_x{x}_y{y}.png'))
plt.show()
plt.close()
#%%
loss_log = os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_loss_values_step.txt')

optimizer = optim.Adam([prior_model_inputs_leaf], lr=learning_rate)

fig, ax = plt.subplots()

for step in range(num_steps):
    print(f'Step {step} of {num_steps}')
    t1 = default_timer()
    optimizer.zero_grad()
    pred = model(prior_model_inputs)
    pred_un = y_normalizer.decode(pred)[0, :, x, y, 0]

    if regularization_weight > 0.0:
        loss = F.mse_loss(observed, pred_un, reduction='mean') + regularization_weight * torch.norm(
            prior_model_inputs[:, :, :, UNKNOWN_PARAMETERS]) ** 2
    else:
        loss = F.mse_loss(observed, pred_un, reduction='mean')

    loss.backward()
    optimizer.step()

    predicted_values.append(pred_un.detach().numpy())
    parameter_values.append(prior_model_inputs[0, -1, :, :, UNKNOWN_PARAMETERS].detach().numpy())
    loss_values.append(loss.item())
    t2 = default_timer()

    print(f'ep {step}: t={t2 - t1:.3f}, mse={loss.item():.3e}')
    with open(loss_log, 'a') as f:
        f.write(f'epoch {step}: t={t2 - t1:.3f}, mse={loss.item():.3e}\n')

    if step % 10 == 0:
        fig, ax = plt.subplots(ncols=2, nrows=1)
        ax[0].imshow(true_map.detach().numpy(), cmap='jet')
        ax[0].set_title(f'Reference permeability values')
        ax[1].imshow(prior_model_inputs[0, -1, :, :, UNKNOWN_PARAMETERS].detach().numpy(), cmap='jet')
        ax[1].set_title(f'Initial permeability values')
        plt.savefig(os.path.join(results_folder, f'Permeability_{step}.png'))
        plt.show()
        plt.close()

        fig, main_ax = plt.subplots()

        main_ax.plot(time, observed, color='red', label='Reference case - true')
        main_ax.plot(time, pred_un.detach().numpy(), color='blue', linestyle='--', label='Prior case - FNO')
        main_ax.plot(time, y_normalizer.decode(pred).detach().numpy()[0, :, x, y, 0], color='green', linestyle='--', label='Posterior case - FNO')
        main_ax.legend()
        main_ax.set_xlabel('Time')
        main_ax.legend()
        main_ax.set_xlabel('Time')
        main_ax.set_ylabel(variable)
        main_ax.set_title(f'Montintoring {variable} at x={x} and y={y}')

        plt.savefig(os.path.join(results_folder, f'Initial_overview_prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_step{step}_used_loss.png'))
        plt.show()
        plt.savefig(os.path.join(image_folder, f'Optimim_{step}.png'))
        plt.close()

        with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_predicted_values_step{step}.pkl'), 'wb') as f:
            pickle.dump(predicted_values, f)
        with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_permeability_values_step{step}.pkl'), 'wb') as f:
            pickle.dump(parameter_values, f)
        with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_loss_values_step{step}.pkl'), 'wb') as f:
            pickle.dump(loss_values, f)
#%%
with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_predicted_values_step{step}.pkl'), 'wb') as f:
    pickle.dump(predicted_values, f)
with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_permeability_values_step{step}.pkl'), 'wb') as f:
    pickle.dump(parameter_values, f)
with open(os.path.join(results_folder, f'prior_{prior_model}_reference_{reference_model}_x{x}_y{y}_posterior_loss_values_step{step}.pkl'), 'wb') as f:
    pickle.dump(loss_values, f)

# %%
