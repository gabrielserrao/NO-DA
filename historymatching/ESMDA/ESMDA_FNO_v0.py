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

NeHf = 1 #ensemble members for the High Fidelity foward model
NePx = 300 #ensemble members for the Proxy
Ne = NeHf + NePx #number of ensemble members

# %%
#----DEFINE THE PRIOR ENSEMBLE----#
prior_path = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/simulations/it0/geo'
 
MGridPrior = np.empty([Nm, Ne])
for i, file in enumerate(os.listdir(prior_path)):
    if i < Ne:
        realization = xr.open_dataset(os.path.join(prior_path,file))
        Permeability = realization['Perm'].values.flatten()
        #apply the log transform to the permeability
        Permeability = np.log(Permeability)
        Porosity = realization['Por'].values.flatten()
        MGridPrior[:,i] = np.concatenate((Permeability, Porosity))







# #plot the prior ensemble for the first case 
# fig, ax = plt.subplots(1,1, figsize=(10,10))
# #split MPrior into permeability and porosity for this jump the number of grid cells
# plt.imshow(MPrior[:int(NGrid/2), 0].reshape((Ni,Nj),order='F').T, cmap='RdYlGn_r', aspect='auto')
# plt.colorbar()
# plt.show()


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
MGrid = MGridPrior
MScalar = []
ESMDA_its = 2
alphas=np.ones(ESMDA_its)*ESMDA_its
MObj=np.zeros([len(alphas),Ne])
#Run
def compute_observation_matrix(directory_path, monitoring_positions):
    D = []
    # Loop over all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file is a .nc file
        if filename.endswith(".nc"):
            # Load the .nc file
            model = xr.open_dataset(os.path.join(directory_path, filename))
            # Extract observation data from the model
            obsData = []
            for (i, j) in monitoring_positions:
                obsData.append(model['Pressure'].isel(X=i).isel(Y=j).values)
            obsValues = np.array(obsData)

            # Flatten the observation values and append them to the D matrix
            D.append(obsValues.flatten())

    # Convert D to a numpy array
    D = np.array(D)

    return D
#%%
start= time.time()
l = 0
for alpha in alphas:
    # Generates the perturbed observations (10.27)
    z = np.random.normal(size=(Nd, Ne))
    DPObs = dObs[:, np.newaxis] + math.sqrt(alpha) * CeDiag[:, np.newaxis] * z
    # 2. Forecast
    destDir= os.path.join(curDir,f'simulations/it{l}')
    geoDir= os.path.join(curDir,f'simulations/it{l}/geo')
    dynDir= os.path.join(curDir,f'simulations/it{l}/dyn')
    if l==0:
        #Define M matrix
        MGrid = MGridPrior
        #compute D Matrix

        if not os.path.exists(dynDir):
            os.makedirs(dynDir)
        if not os.path.exists(geoDir):
            os.makedirs(geoDir)        
        for i in range(Ne):
            if i < NeHf:
                run_forward(reference_folder,
                            data_folder = geoDir,
                            numberHFmembers =NeHf,
                            output_folder=dynDir,
                            is_proxy = False,
                            )
            else:
                run_forward(reference_folder,
                            data_folder = geoDir,
                            numberHFmembers=NeHf,
                            output_folder=dynDir,
                            is_proxy = True,
                            )
        #%%
        D = compute_observation_matrix(dynDir, monitoring_positions)
        pd.DataFrame(D).to_pickle(f'{destDir}/D_{l}.pkl')

                
        #%%
        llllk
        #read the results of the forward model  
  
        #compute the objective function
        MObj[l,:]=calcDataMismatchObjectiveFunction(dObs[:,np.newaxis], D, CeInv)
        pd.DataFrame(MObj[l,:]).to_pickle(f'{destDir}/MObj_{l}.pkl')
        #compute the mean of the objective function
        MObjMean = np.mean(MObj, axis=1)
            

        prior_path = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/prior_geomodels'
        
    
    destDir= os.path.join(curDir,f'simulations/it{l}')
    #check if the directory exists, if not create it
    if not os.path.exists(destDir):
        os.makedirs(destDir)
    #compute the forward model for the ensemble members
    run_forward(reference_folder,
                data_folder = destDir,
                numberHFmembers,
                output_folder=destDir,
                is_proxy = False,
                ):  

  

    
    

   
    # Run the simulations g(M) (12.4)
    mGridHf mGrid[:NeHf]     
    inputParamsHf['Permeability'] = mGridHf
    RunDarts(inputParamsHf, destDirHF, l)
    filename = os.path.join(destDir,f'data_model'+str(Ne-1)+'.pkl')
    check_job(filename)
    D_Hf =ReadHFResults(destDirHF,l)
    pd.DataFrame(D_Hf).to_pickle(f'{destDir}/D_{l}.pkl')

    mGridPx = mGrid[:NePx]     
    inputParamsPx['Permeability'] = mGridPx
    RunProxy(inputParamsPx, destDirPx, l)
    filename = os.path.join(destDir,f'data_model'+str(Ne-1)+'.pkl')
    check_job(filename)
    D_Px =ReadPxResults(destDirPx,l)
    pd.DataFrame(D_Px).to_pickle(f'{destDir}/D_{l}.pkl')

    D = np.concatenate([D_Hf, D_Px])
    pd.DataFrame(D_Px).to_pickle(f'{destDir}/D_{l}.pkl')
    
    if (l == 0):
        DPrior = D

    DobsD = DPObs - D

    # 4. Analysis
    # 4.1 Invert matrix C
    # Calculates DeltaD (12.5)
    meanMatrix = np.mean(D, axis=1)
    DeltaD = D - meanMatrix[:, np.newaxis]
    # Calculates CHat (12.10)
    CHat = SInvDiag[:, np.newaxis] * \
        (DeltaD @ DeltaD.T) * \
        SInvDiag[np.newaxis, :] + alpha * (Ne - 1) * INd

    # Calculates Gamma and X (12.18)
    U, SigmaDiag, Vt = np.linalg.svd(CHat)
    Nr = FindTruncationNumber(SigmaDiag, csi)

    GammaDiag = np.power(SigmaDiag[0:Nr], -1)
    X = SInvDiag[:, np.newaxis] * U[:, 0:Nr]

    # Calculates M^a (12.21)
    X1 = GammaDiag[:, np.newaxis] * X.T
    X8 = DeltaD.T @ X
    X9 = X8 @ X1

    # subpart: for grid, use localization
    #MGrid = UpdateModelLocalized(MGrid, X9, Rmd, DobsD)
    #test without localization
    MGrid = UpdateModel(MGrid, X9, DobsD)
    pd.DataFrame(MGrid).to_pickle(f'{destDir}/MGrid_{l}.pkl')
   
    CeInv = np.power(CeDiag, -1)
    MObj[l,:]=calcDataMismatchObjectiveFunction(dObs[:,np.newaxis], D, CeInv)
    pd.DataFrame(MObj[l,:]).to_pickle(f'{destDir}/MObj_{l}.pkl')
    
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

           
# # %%
# plt.imshow(MPrior[int(NGrid/2):, 0].reshape((Ni,Nj),order='F').T, cmap='RdYlGn_r', aspect='auto')
# plt.show()

# %%
