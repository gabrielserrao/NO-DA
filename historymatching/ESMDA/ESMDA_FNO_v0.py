 # %%
import os
import numpy as np
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
from runForward import run_forward
from utilsESMDA import  ModelOut, deleteOutFiles, media_function, get_grid_outputs
from utilsESMDA import  CalcHL, SphereFunction, GaspariCohnFunction, IndexToIJ, IJToIndex, BuildPermCovMatrix, \
    BuildLocalizationMatrix, PlotModelRealization, PlotMatrix, RunModels, check_job, ReadModels, \
        FindTruncationNumber, CentralizeMatrix, UpdateModelLocalized, UpdateModel, \
            calcDataMismatchObjectiveFunction, MultipliNegatives


# %% [markdown]
 #read reference metadata from reference_folder
reference_folder = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/REFERENCE'
with open(os.path.join(reference_folder,'Reference_metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)

reference_model = xr.open_dataset(os.path.join(reference_folder,'ReferenceSimulation.nc'))

dt = metadata.iloc[0].values[0]
nsteps = metadata.iloc[1].values[0]
well_coords = metadata.iloc[2].values[0]
well_rates = metadata.iloc[3].values[0]
initial_gas_rate = metadata.iloc[4].values[0]
# output_folder = metadata.iloc[5].values[0]
# output_filename = metadata.iloc[6].values[0]
treatGeoModel = metadata.iloc[7].values[0]
RefGeoData_path = metadata.iloc[8].values[0]
#%%
#----DEFINE OBSERVATION DATA----#
#define grid locations of the monitoring points on the 32x32 grid (for all time steps)
monitoring_numbers = 4
monitoring_positions = [[2,2], [18,2] ,[2,18], [18,18]] #wells position in the 20x20 grid

obsData = []
for (i,j) in monitoring_positions:
    obsData.append(reference_model['Pressure'].isel(X=i).isel(Y=j).values)
obsValues=np.array(obsData)
dObs = obsValues.flatten()
CeDiag =np.where(np.array(0.05*dObs[:])<1e-3, 1e-3, 0.01*dObs[:])

time_range = reference_model.time.values
NTimesteps=len(time_range)


#wellDObs = np.repeat(wells, NTimesteps) # Configure the wells list

# %% [markdown]
grid_variables = 2 #number of variables in the grid for the DA (permeability, porosity)
Ni = len(reference_model.X.values)
Nj = len(reference_model.X.values)
NGrid = Ni * Nj * grid_variables
NScalar = 0 #we are not considering any scalar parameters in the problem like kro, krw 
Nm = NGrid + NScalar
Nd = len(dObs)  #len(dTime)* obsValues.shape[0] #  timesteps * 4 well datas

NeHf = 100 #ensemble members for the High Fidelity foward model
NePx = 900 #ensemble members for the Proxy
Ne = NeHf + NePx #number of ensemble members


#%% 
#------BUILD LOCALIZATION MATRIX------# 
# Covariogram parameters 
L = (25,25) 
theta = 45 * np.pi/180 #degrees
sigmaPr2 = 1.0
# Localization parameters
locL = (40,20)
locTheta = 45 * np.pi/180
# svd truncation parameter for SVD 
csi = 0.99
#Rmd = BuildLocalizationMatrix(Ni, Nj, wellPos, locL, locTheta)

#%%
#------RUN DATA ASSIMILATION------# 
#Define directories
curDir = os.getcwd()
srcDir =  f'{curDir}'

#Define the error matriz for the analyses step
SDiag = np.sqrt(CeDiag)
SInvDiag = np.power(SDiag, -1)
INd = np.eye(Nd)

MScalar = []
ESMDA_its = 4
alphas=np.ones(ESMDA_its)*ESMDA_its
MObj=np.zeros([len(alphas),Ne])
#Run

def compute_observation_matrix(directory_path, Ne, monitoring_positions):
    D = []

    # Loop over all proxy files in the directory
    for i in range(Ne):
        # If the corresponding HF file exists, load that instead
        if os.path.isfile(os.path.join(directory_path, f"darts_out_{i}.nc")):
            model = xr.open_dataset(os.path.join(directory_path, f"darts_out_{i}.nc"))
        else:
            model = xr.open_dataset(os.path.join(directory_path, f"proxy_out_{i}.nc"))

        # Extract observation data from the model
        obsData = [model['Pressure'].isel(X=x, Y=y).values for x, y in monitoring_positions]

        # Flatten the observation values and append them to the D matrix
        D.append(np.array(obsData).flatten())
    
    D = np.array(D)
    D = D.T # Nobs x Ne

    return D

#%%
start= time.time()
l = 0
for alpha in alphas:
    # Generates the perturbed observations (10.27)
    z = np.random.normal(size=(Nd, Ne))
    DPObs = dObs[:, np.newaxis] + math.sqrt(alpha) * CeDiag[:, np.newaxis] * z
    
    prior_path = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/prior_geomodels'

    destDir= os.path.join(curDir,f'simulations/it{l}')
    geoDir= os.path.join(destDir,f'geo')
    dynDir= os.path.join(destDir,f'dyn')  
    if not os.path.exists(destDir):
        os.makedirs(destDir)  
    if not os.path.exists(dynDir):
        os.makedirs(dynDir)
    if not os.path.exists(geoDir):
        os.makedirs(geoDir) 
    

    if l==0: 
        MGridPrior = np.empty([Nm, Ne])        
        for i, file in enumerate(os.listdir(prior_path)):
            if i < Ne:
                realization = xr.open_dataset(os.path.join(prior_path,file))
                realization.to_netcdf(f'{geoDir}/geo_{i}.nc')
                Permeability = realization['Perm'].values.flatten()
                #apply the log transform to the permeability
                Permeability = np.log(Permeability)
                Porosity = realization['Por'].values.flatten()
                MGridPrior[:,i] = np.concatenate((Permeability, Porosity))
                #save MGridPrior to pickle
                pd.DataFrame(MGridPrior).to_pickle(f'{destDir}/MGrid_{l}.pkl')
                MGrid = MGridPrior
                

         
        run_forward(reference_folder,
            data_folder = geoDir,
            numberHFmembers =NeHf,
            Ne = Ne,
            output_folder=dynDir,
            is_proxy = False,
            )
        #runs proxy
        run_forward(reference_folder,
                    data_folder = geoDir,
                    numberHFmembers=NeHf,
                    Ne = Ne,
                    output_folder=dynDir,
                    is_proxy = True,
                    )
        
    else:
    #read MGrid from the previous iteration
        destDir= os.path.join(curDir,f'simulations/it{l-1}')
        MGrid = pd.read_pickle(f'{destDir}/MGrid_{l-1}.pkl') 
        MGrid = MGrid.values
        destDir= os.path.join(curDir,f'simulations/it{l}')  
        geoDir= os.path.join(destDir,f'geo')
        dynDir= os.path.join(destDir,f'dyn')    
        for i, file in enumerate(os.listdir(prior_path)):
            if i < Ne:
                print(i)
                Permeability = MGrid[:int(NGrid/2),i]
                Porosity = MGrid[int(NGrid/2):,i]                
                Permeability = np.exp(Permeability)
                Permeability = Permeability.reshape((Ni,Nj),order='F')
                Porosity = Porosity.reshape((Ni,Nj),order='F')
                #overwrite the values of the permeability and porosity in the realization
                realization = xr.open_dataset(os.path.join(prior_path,file))
                realization['Perm'].values = Permeability
                realization['Por'].values = Porosity
                realization.to_netcdf(f'{geoDir}/geo_{i}.nc')

                print('saveed geomodel to netcdf')
            
        print('Running...')

        #runs DARTS
        run_forward(reference_folder,
            data_folder = geoDir,
            numberHFmembers =NeHf,
            Ne = Ne,
            output_folder=dynDir,
            is_proxy = False,
            )
        #runs proxy
        run_forward(reference_folder,
                    data_folder = geoDir,
                    numberHFmembers=NeHf,
                    Ne = Ne,
                    output_folder=dynDir,
                    is_proxy = True,
                    )
    D = compute_observation_matrix(dynDir, Ne, monitoring_positions)
    pd.DataFrame(D).to_pickle(f'{destDir}/D_{l}.pkl')    
    DobsD = DPObs - D
    
    meanMatrix = np.mean(D, axis=1)
    
    DeltaD = D - meanMatrix[:, np.newaxis]

    CHat = SInvDiag[:, np.newaxis] * \
        (DeltaD @ DeltaD.T) * \
        SInvDiag[np.newaxis, :] + alpha * (Ne - 1) * INd

    U, SigmaDiag, Vt = np.linalg.svd(CHat)
    Nr = FindTruncationNumber(SigmaDiag, csi)

    GammaDiag = np.power(SigmaDiag[0:Nr], -1)
    X = SInvDiag[:, np.newaxis] * U[:, 0:Nr]   
    X1 = GammaDiag[:, np.newaxis] * X.T
    X8 = DeltaD.T @ X
    X9 = X8 @ X1

    # subpart: for grid, use localization
    #MGrid = UpdateModelLocalized(MGrid, X9, Rmd, DobsD)
    #test without localization
    MGrid = UpdateModel(MGrid, X9, DobsD)
    pd.DataFrame(MGrid).to_pickle(f'{destDir}/MGrid_{l}.pkl')
 
    CeInv = np.power(CeDiag, -1)  
    #compute the objective function
    MObj[l,:]=calcDataMismatchObjectiveFunction(dObs[:,np.newaxis], D, CeInv)
    pd.DataFrame(MObj[l,:]).to_pickle(f'{destDir}/MObj_{l}.pkl')

    #compute the mean of the objective function
    MObjMean = np.mean(MObj, axis=1)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(MObjMean, color='r', alpha=0.8)
    ax.set_title('Objective function')
    ax.set_xlabel('Iteration')
    ax.set_yscale('log')
    ax.set_ylabel(f'Mean of Objective function until ES-MDA iteration {l}')
    fig.savefig(f'{destDir}/Mean_of_Objective_function_{l}.jpg')  

    l += 1

MGridPost = MGrid
MScalarPost = MScalar
DPost = D

end = time.time()
elapsed = end - start
print('Elapsed time of the ES-MDA: ', elapsed)

           

# %%
