         

# %%
import sys
sys.path.append("..")
sys.path.append(".")

import torch
import os
from utilities import *
from model_fourier_3d import *
from torch.utils.data import DataLoader
import re

import math
import os
import re
import shutil
import time 
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import glob
import xarray as xr
from runForward_original import run_forward
from utilsESMDA import  ModelOut, deleteOutFiles, media_function, get_grid_outputs
from utilsESMDA import  CalcHL, SphereFunction, GaspariCohnFunction, IndexToIJ, IJToIndex, BuildPermCovMatrix, \
    BuildLocalizationMatrix, PlotModelRealization, PlotMatrix, RunModels, check_job, ReadModels, \
        FindTruncationNumber, CentralizeMatrix, UpdateModelLocalized, UpdateModel, \
            calcDataMismatchObjectiveFunction, MultipliNegatives, compute_observation_matrix

from torch import optim
from datetime import datetime
# %% [markdown]
###############################
# LOAD REFERENCE MODEL AND CALCULATE OBSERVATION VECTOR AND ERROR
###############################
REFERENCE_FOLDER = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/REFERENCE'
MONITORING_POSITIONS = [[11, 11], [11, 21], [21, 11], [21, 21]] 
OPTIMIZATION_STEPS =11
NUM_EMSEMBLES = 5
BASE_PATH = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/VARIATIONAL/'
RUN_PATH = os.path.join(BASE_PATH, f'runs/{OPTIMIZATION_STEPS}steps_{NUM_EMSEMBLES}ensembles')
os.makedirs(RUN_PATH, exist_ok=True)
# Function to load reference model and metadata
def load_reference_data(folder_path):
    with open(os.path.join(folder_path, 'Reference_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    reference_model = xr.open_dataset(os.path.join(folder_path, 'ReferenceSimulation.nc'))
    
    return metadata, reference_model

# Function to calculate observation data and error
def calculate_observation_data(reference_model, monitoring_positions):
    obs_values = np.array([reference_model['Pressure'].isel(X=i, Y=j).values for i, j in monitoring_positions])
    obs_values = np.array([obs_values[i][0:] for i in range(len(obs_values))])
    d_obs = obs_values.flatten()
    CeDiag = np.where(0.05 * d_obs < 1e-3, 1e-3, 0.01 * d_obs)
    
    return d_obs, CeDiag

def generate_perturbed_observations(d_obs, CeDiag, num_realizations=100):
    # Create a list to hold all the perturbed observation realizations
    dObse = []
    for _ in range(num_realizations):
        # Generate random noise for each observation by sampling from a normal distribution
        noise = np.random.normal(0, CeDiag)
        # Add the noise to the original observation data to create a perturbed realization
        perturbed_observation = d_obs + noise
        dObse.append(perturbed_observation)
    return np.array(dObse)

metadata, reference_model = load_reference_data(REFERENCE_FOLDER)
dt, nsteps, well_coords, well_rates, initial_gas_rate, _ , _ , treatGeoModel, RefGeoData_path = metadata.iloc[:9, 0]

dObs, CeDiag = calculate_observation_data(reference_model, MONITORING_POSITIONS)
time_range = reference_model.time.values
NTimesteps = len(time_range)
list_pos = [pos for pos in MONITORING_POSITIONS for _ in range(NTimesteps)]


# %% [markdown]
###############################
# DEFINE PRIOR
###############################
""" 
Prior model is the model that will be used as initial guess for the method.
We need to load the priors geomodels genetared off line and, APART OF THE OPTIMIZATION VARIABLE, we change all the properties to the same as the reference model.  
"""

Ne = NUM_EMSEMBLES # number of ensemble members
grid_variables = 1 #number of variables in the grid for the DA (permeability,)
Ni = len(reference_model.X.values)
Nj = len(reference_model.X.values)
NGrid = Ni * Nj * grid_variables
NScalar = 0 #we are not considering any scalar parameters in the problem like kro, krw 
Nm = NGrid + NScalar
Nd = len(dObs) 
dObse = generate_perturbed_observations(dObs, CeDiag, num_realizations=Ne)

RUN_PRIOR = True
 #where to get the geomodels permeability field

curDir = RUN_PATH
geo_prior_Dir= os.path.join(curDir,f'geomodel_prior') 
darts_prior_Dir= os.path.join(curDir,f'dynmodel_darts_prior')
geomodel_post_optm_Dir  = os.path.join(curDir,f'geomodel_post_optm')
dynmodel_darts_posterior_Dir = os.path.join(curDir,f'dynmodel_darts_posterior')
#check if the folders exist, if not create them
directories = [geo_prior_Dir, darts_prior_Dir, geomodel_post_optm_Dir, dynmodel_darts_posterior_Dir]
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)


MGridPrior = np.empty([Nm, Ne])        


if RUN_PRIOR:
        GEOMODELS_PATH = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/prior_geomodels'
        for i, file in enumerate(os.listdir(GEOMODELS_PATH)):
            if i < Ne:
                realization = xr.open_dataset(os.path.join(GEOMODELS_PATH,file))
                realization.to_netcdf(f'{geo_prior_Dir}/geo_{i}.nc')
                Permeability = realization['Perm'].values.flatten()
                Permeability[Permeability<0] = 0.01 
    
        run_forward(REFERENCE_FOLDER,
        data_folder = geo_prior_Dir,
        numberHFmembers = Ne,
        Ne = Ne,
        output_folder=darts_prior_Dir,
        is_proxy = False,
        )
else:
    """
    If we are not running the prior, we need to paste runned simulations on the darts_prior_Dir folder
    Folder  '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/prior_dynmodels/ensemble_100' already have the prior simulationsfor the current reference case 
    PRIOR NEEDS TO BE CHANGED IF REFERENCE CASE IS CHANGED DUE TO PARAMETERS THAT ARE NOT TREATED AS OPTIMIZATION VARIABLES
    """ 
    darts_prior_Dir = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/VARIATIONAL/dynmodel_darts_prior' 
    for i, file in enumerate(os.listdir(darts_prior_Dir)[:Ne]):
        if i < Ne:
            realization = xr.open_dataset(os.path.join(darts_prior_Dir,file))
            Permeability = realization['Perm'].values.flatten()
            Permeability[Permeability<0] = 0.01     
            MGridPrior[:,i] = Permeability #np.concatenate((Permeability, Porosity))

#save MGridPrior to pickle
pd.DataFrame(MGridPrior).to_pickle(f'{curDir}/MGridPrior.pkl')
MGrid = MGridPrior  
#Store DPrior, read all .nc on darts_prior_Dir, read the observation vector and append to DPrior
DPrior = np.empty([Nd, Ne])
for i, file in enumerate(os.listdir(darts_prior_Dir)):
    if i < Ne:
        realization = xr.open_dataset(os.path.join(darts_prior_Dir,file))     
        DPrior[:,i], _ = calculate_observation_data(realization, MONITORING_POSITIONS)
        #save DPrior to pickle
        pd.DataFrame(DPrior).to_pickle(f'{curDir}/DPrior.pkl')

#%%

###############################
# RUN OPTIMIZATION WITH PROXY MODEL
###############################
PATH_NN_MODEL = '/samoa/data/smrserraoseabr/NO-DA/runs/FNO_3d_MonthQgWellCenter_N800.0_ep100_m18_w128_b1_normPointGaussianNormalizer_INPUT_Por_Perm_gas_rate_OUTPUT_Pressure/FNO_3d_MonthQgWellCenter_N800.0_ep100_m18_w128_b1_normPointGaussianNormalizer_INPUT_Por_Perm_gas_rate_OUTPUT_Pressure_model.pt'
INPUT_VARS = ['Por', 'Perm', 'gas_rate']
OUTPUT_VARS = ['Pressure']
WELLS_POSITIONS = True
DEVICE = 'cuda'
output_folder = curDir
# List all files in the original data_folder
all_files = sorted(glob.glob(os.path.join(geo_prior_Dir, '*.nc')))


folder = os.path.dirname(PATH_NN_MODEL)       
batch_size = 1  # Use batch size of 1
num_files = Ne
dataset = ReadXarrayDataset(folder=darts_prior_Dir, 
                        input_vars=INPUT_VARS, 
                        output_vars=OUTPUT_VARS,
                        num_files = num_files, 
                        wells_positions=WELLS_POSITIONS
                        )

data_loader = DataLoader(dataset, 
                    batch_size=batch_size,
                    shuffle=False) 

path_normalizer_mean_input = os.path.join(folder, 'normalizer_mean_input.pt')
path_normalizer_std_input = os.path.join(folder, 'normalizer_std_input.pt')
path_normalizer_mean_output = os.path.join(folder, 'normalizer_mean_output.pt')
path_normalizer_std_output = os.path.join(folder, 'normalizer_std_output.pt')   
input_normalizer_mean = torch.load(path_normalizer_mean_input)
input_normalizer_std = torch.load(path_normalizer_std_input)
output_normalizer_mean = torch.load(path_normalizer_mean_output)
output_normalizer_std = torch.load(path_normalizer_std_output) 

input_normalizer = PointGaussianNormalizer(data_loader, 
                                        mean = input_normalizer_mean, 
                                        std = input_normalizer_std, 
                                        is_label=False)

output_normalizer = PointGaussianNormalizer(data_loader, 
                                        mean = output_normalizer_mean,
                                        std = output_normalizer_std, 
                                        is_label=True)

input_normalizer = input_normalizer.cuda(DEVICE)
output_normalizer = output_normalizer.cuda(DEVICE)

model = torch.load(PATH_NN_MODEL, map_location=DEVICE)
model.to(DEVICE)
model.eval()
global_count = 0  # Global file counter
#%%
###############################
#COMPUTE PROXY PRIOR RESULT
###############################
DProxyPrior = np.empty([Nd, Ne])
DProxy = np.empty([Nd, Ne])
DPost = np.empty([Nd, Ne])
MGrid = np.empty([Nm, Ne])
for i, (x, _) in enumerate(data_loader):
    time = x.shape[0]
    Y = x.shape[1]
    X = x.shape[2]
    x = x.to(DEVICE)
    x = input_normalizer.encode(x)
    out = model(x) #[batch, time, Y, X, output_size]
    out = output_normalizer.decode(out)
    out = out.detach().cpu().numpy()    
    for o in out:
        DProxyPrior[:,global_count] = np.array([o[:, i, j, 0] for i, j in MONITORING_POSITIONS]).flatten()        
        
    for m in x:
        proxy_model_inputs = input_normalizer.decode(m.unsqueeze(0))[0, :, :, :, :].unsqueeze(0)
        prior_model_inputs_PARAM_leaf = torch.tensor(proxy_model_inputs[:, :, :, :, 1], requires_grad=True).to(DEVICE)
        optimizer = optim.Adam([prior_model_inputs_PARAM_leaf], lr=0.01)
        num_steps = OPTIMIZATION_STEPS
        for step in range(num_steps):
            optimizer.zero_grad()
            proxy_model_inputs[:, :, :, :, 1].unsqueeze(-1).copy_(prior_model_inputs_PARAM_leaf.unsqueeze(-1))     
     
            # Compute the model's output (assuming 'out' is the model's output)
            out_m = model(input_normalizer.encode(proxy_model_inputs))
            #decode the output
            out_m = output_normalizer.decode(out_m)
            
                    # Extract the predicted values at the monitoring positions
            dProxy = [out_m[0, :, i, j, 0] for (i, j) in MONITORING_POSITIONS]
            predicted_tensor = torch.stack(dProxy).flatten().to(DEVICE)

            observed_tensor = torch.tensor(dObse[global_count,:]).to(DEVICE)
            loss = torch.sum((predicted_tensor - observed_tensor) ** 2)
            # Backpropagate
            loss.backward(retain_graph=True)
            optimizer.step()
            prior_model_inputs_PARAM_leaf.data.clamp_(min=0)            

            DProxy[:, global_count] = np.array([out_m[0, :, i, j, 0].cpu().detach().numpy() for (i, j) in MONITORING_POSITIONS]).flatten()

            
            print(f'######## Model {global_count} ##########')   
            print('step: %d, loss: %.3e' % (step, loss.item()))         
            if step % 10 == 0:
                # print('step: %d, loss: %.3e' % (step, loss.item()))                # plt.plot(dObs, label='observed', marker='o', linestyle='none')
                # plt.plot(DProxy[:,global_count].flatten(), label='predicted')
                # plt.plot(DProxyPrior[:,global_count].flatten(), label='prior')
                # plt.legend()
                # plt.show()
                posterior_model_inputs = prior_model_inputs_PARAM_leaf.detach().cpu().numpy()
                #save the posterior model inputs
     
        PosteriorInputs = input_normalizer.decode(proxy_model_inputs).detach().cpu().numpy()[0,0,:,:,1]         
        MGrid[:, global_count] = PosteriorInputs.flatten()       
        DProxy[:,global_count] = DProxy[:,global_count].flatten()  
        pd.DataFrame(DProxy).to_pickle(f'{curDir}/DPost_Proxy.pkl')      
        DProxyPrior[:,global_count] = DProxyPrior[:,global_count].flatten()       
        global_count += 1  
#%%
pd.DataFrame(DProxy).to_pickle(f'{curDir}/DPost_Proxy.pkl') 
#Run DARTS with posterior models
pd.DataFrame(MGrid).to_pickle(f'{curDir}/MGrid.pkl')
for i in range(Ne):  
    realization = xr.open_dataset(os.path.join(darts_prior_Dir,file))                
    Permeability = MGrid[:, i]
    Permeability = Permeability.reshape((Ni,Nj),order='F')
    Permeability[Permeability<0.01] = 0.01                
    realization['Perm'].values = Permeability       
    realization.to_netcdf(f'{geomodel_post_optm_Dir}/geo_{i}.nc')

    
    
    
run_forward(REFERENCE_FOLDER,
data_folder = geomodel_post_optm_Dir,
numberHFmembers = Ne,
Ne = Ne,
output_folder= dynmodel_darts_posterior_Dir,
is_proxy = False,
)

for i, file in enumerate(os.listdir(dynmodel_darts_posterior_Dir)):
    if i < Ne:
        realization = xr.open_dataset(os.path.join(dynmodel_darts_posterior_Dir,file))     
        DPost[:,i], _ = calculate_observation_data(realization, MONITORING_POSITIONS)
        #save DPrior to pickle
        pd.DataFrame(DPost).to_pickle(f'{curDir}/DPost_DARTS.pkl')
#%%
plt.plot(dObs, label='observed', marker='o', linestyle='none')
plt.plot(DPost, label='DARTS posterior')
#plt.plot(DPrior, label='DARTS prior')
plt.legend()
plt.show()
#%%
plt.plot(dObs, label='observed', marker='o', linestyle='none')
#plt.plot(DPost, label='DARTS posterior')
plt.plot(DPrior, label='DARTS prior')
plt.legend()
plt.show()
        
             
                
# #Calculate the objective function
# MObj[l,:]=calcDataMismatchObjectiveFunction(dObs[:,np.newaxis], D, CeInv)

# Od = np.empty(Ne)
# for i in range(Ne):
#     dObsD = dObs - D[:,i].reshape(Nd,1)
#     Od[i] = (dObsD.T) @ (CeInv[:,np.newaxis] * dObsD)/2

# %%
                       
        ###############################      
        # RUN OPTIMIZATION WITH PROXY MODEL
        ###############################
        

        #compare DProxyPrior with DPrior
        # plt.figure()
        # plt.plot(dObs, label='observed', marker='o', linestyle='none')
        # plt.plot(DPrior.flatten(), label='DARTS Prior')
        # plt.plot(DProxyPrior.flatten(), label='Proxy Prior')
        # plt.legend()
#%%
y_limits=[200, 320]
Nd=61
filter = 0
DPrior_reshaped = DPrior.reshape((4, Nd, -1))[:, :, :Ne]
dObs_reshaped = dObs.reshape((4, Nd))
DPosterior_reshaped = DPost.reshape((4, Nd, -1))[:, :, :Ne]
CeDiag_reshaped = CeDiag.reshape((4, Nd))
time_range = time_range[filter:]
fig, axes = plt.subplots(4, 2, figsize=(15, 20), sharey=True)
for i, (ax_prior, ax_posterior) in enumerate(axes):
    # Observed Data
    ax_prior.errorbar(time_range, dObs_reshaped[i], yerr=CeDiag_reshaped[i], fmt='o', color='r', label='Observed Data')
    ax_posterior.errorbar(time_range, dObs_reshaped[i], yerr=CeDiag_reshaped[i], fmt='o', color='r', label='Observed Data')
    
    # Prior Individual Curves
    for realization in DPrior_reshaped[i, :, :].T:
        ax_prior.plot(time_range, realization, color='gray', alpha=0.3)

    # Posterior Individual Curves
    for realization in DPosterior_reshaped[i, :, :].T:
        ax_posterior.plot(time_range, realization, color='blue', alpha=0.3)

    # Prior Statistics
    prior_p10 = np.percentile(DPrior_reshaped[i, :, :], 10, axis=1)
    prior_p90 = np.percentile(DPrior_reshaped[i, :, :], 90, axis=1)
    prior_mean = DPrior_reshaped[i, :, :].mean(axis=1)
    #ax_prior.fill_between(time_range, prior_p10, prior_p90, color='gray', alpha=0.5, label='Prior (P10-P90)')
    #ax_prior.plot(time_range, prior_mean, 'k-', lw=2, label='Prior Mean')

    # Posterior Statistics
    posterior_p10 = np.percentile(DPosterior_reshaped[i, :, :], 10, axis=1)
    posterior_p50 = np.percentile(DPosterior_reshaped[i, :, :], 50, axis=1)
    posterior_p90 = np.percentile(DPosterior_reshaped[i, :, :], 90, axis=1)
    #ax_posterior.fill_between(time_range, posterior_p10, posterior_p90, color='blue', alpha=0.5, label='Posterior (P10-P90)')
    #ax_posterior.plot(time_range, posterior_p50, 'k-', lw=2, label='Posterior Mean')

    ax_prior.set_ylim(y_limits)
    ax_prior.set_title(f'Monitoring Point {i+1} - Prior')
    ax_posterior.set_title(f'Monitoring Point {i+1} - Posterior')
    ax_prior.set_ylabel('Pressure (bar)')
    ax_prior.legend(loc='upper left')
    ax_posterior.legend(loc='upper left')
ax_prior.set_xlabel('Time (days)')
ax_posterior.set_xlabel('Time (days)')  
#save figure
plt.savefig(f'{curDir}/HistoryMatching_{Ne}.png', dpi=400)
#save a txt ile with the current time of the job end of the job
with open(f'{curDir}/end_time.txt', 'w') as f:
    #write that the DATA ASSIMILATION WAS FINISHED ON ...add()
    f.write(f'History Matching was finished on {datetime.now()}')
# %%
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# # Create a figure
# fig = plt.figure(figsize=(15, 5))

# # Define a GridSpec layout
# gs = gridspec.GridSpec(1, 6, width_ratios=[1, 0.05, 1, 0.05, 1, 0.05])

# # Plot the Prior
# ax0 = plt.subplot(gs[0])
# im0 = ax0.imshow(MGridPrior[:, 2].reshape((Ni, Nj), order='F'), cmap='jet')
# plt.title('Prior')
# ax0.set_xticks([])
# ax0.set_yticks([])

# # Add colorbar for Prior
# cax0 = plt.subplot(gs[1])
# plt.colorbar(im0, cax=cax0)

# # Plot the Posterior
# ax1 = plt.subplot(gs[2])
# im1 = ax1.imshow(MGrid[:, 2].reshape((Ni, Nj), order='F'), cmap='jet')
# plt.title('Posterior')
# ax1.set_xticks([])
# ax1.set_yticks([])

# # Add colorbar for Posterior
# cax1 = plt.subplot(gs[3])
# plt.colorbar(im1, cax=cax1)

# # Plot the Reference Model
# ax2 = plt.subplot(gs[4])
# im2 = ax2.imshow(reference_model.Perm.values.T.flatten().reshape((Ni, Nj)), cmap='jet')
# plt.title('Reference Model')
# ax2.set_xticks([])
# ax2.set_yticks([])

# # Add colorbar for Reference Model
# cax2 = plt.subplot(gs[5])
# plt.colorbar(im2, cax=cax2)

# plt.tight_layout()
# plt.show()

# #save the figure
# plt.savefig(f'{curDir}/HistoryMatching_{Ne}_Permeablity.png', dpi=300)
# %%
