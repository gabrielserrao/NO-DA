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
            calcDataMismatchObjectiveFunction, MultipliNegatives, compute_observation_matrix

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

#last_data_for_HM =12

monitoring_positions = [[11, 11], [11, 21], [21, 11], [21, 21]]  # Grid locations of the monitoring points on the 32x32 grid (for all time steps)
obsValues = np.array([reference_model['Pressure'].isel(X=i, Y=j).values for i, j in monitoring_positions])
#Remove the first value as it is the same in all the models and collpases the adjust 

obsValues_full = np.array([obsValues[i][:0] for i in range(len(obsValues))])
dObs_full = obsValues_full.flatten()

obsValues = np.array([obsValues[i][:last_data_for_HM] for i in range(len(obsValues))])
dObs = obsValues.flatten()
#CeDiag = np.where(0.05 * dObs < 1e-3, 1e-3, 0.05 * dObs)
CeDiag = np.where(0.05 * dObs < 1e-3, 1e-3, 0.01 * dObs)
time_range = reference_model.time.values
NTimesteps = len(time_range)
# Define a list of tuples of the monitoring_positions position for each dObs for Localization
list_pos = [pos for pos in monitoring_positions for _ in range(NTimesteps)]
#define !export CUDA_VISIBLE_DEVICES=1 on pytorch


# %% [markdown]
grid_variables = 1 #number of variables in the grid for the DA (permeability,)
Ni = len(reference_model.X.values)
Nj = len(reference_model.X.values)
NGrid = Ni * Nj * grid_variables
NScalar = 0 #we are not considering any scalar parameters in the problem like kro, krw 
Nm = NGrid + NScalar
Nd = len(dObs)  #len(dTime)* obsValues.shape[0] #  timesteps * 4 well datas

NeHf = 100 #ensemble members for the High Fidelity foward model
NePx = 0 #ensemble members for the Proxy
l = 0 #which iteration of the ES-MDA are we starting from
Ne = NeHf + NePx #number of ensemble members
ESMDA_its = 8 
RunPrior = False
Localization = None #'Correlation' #Correlation, GaspariCohn, None
path_case = f'datafilter{last_data_for_HM}_REF1000_PARALLEL_simulations_Loc_{Localization}_PERM_Nit{ESMDA_its}_HF{NeHf}_PX{NePx}'
#check if path_case exists
if not os.path.exists(path_case):
    os.makedirs(path_case)
#%% 
#------BUILD LOCALIZATION MATRIX------# 
if Localization == 'Correlation':
    from localization import proxy_corr_localization2
    data_folder ='/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/prior_geomodels'
    path_model = \
        '/samoa/data/smrserraoseabr/NO-DA/runs/FNO_3d_MonthQgWellCenter_N800.0_ep100_m18_w128_b1_normPointGaussianNormalizer_INPUT_Por_Perm_gas_rate_OUTPUT_Pressure/FNO_3d_MonthQgWellCenter_N800.0_ep100_m18_w128_b1_normPointGaussianNormalizer_INPUT_Por_Perm_gas_rate_OUTPUT_Pressure_model.pt'
        #'/samoa/data/smrserraoseabr/NO-DA/runs/FNO_3d_MonthQgWellCenter_HM_N80.0_ep100_m12_w128_b1_normPointGaussianNormalizer_INPUT_Por_Perm_gas_rate_OUTPUT_Pressure/FNO_3d_MonthQgWellCenter_HM_N80.0_ep100_m12_w128_b1_normPointGaussianNormalizer_INPUT_Por_Perm_gas_rate_OUTPUT_Pressure_model.pt'
        
    input_vars = ['Por', 'Perm', 'gas_rate']
    output_vars = ['Pressure']
    WELLS_POSITIONS = True
    device = 'cuda' #change this to gpu 1  # Use GPU
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    output_folder = path_case
    monitoring_positions = monitoring_positions
    
    Rmd = proxy_corr_localization2(data_folder,
                                path_model=path_model, 
                                input_vars=input_vars, 
                                output_vars=output_vars, 
                                device=device, 
                                output_folder=output_folder, 
                                Ne=1000, 
                                monitoring_positions=monitoring_positions,
                                WELLS_POSITIONS=WELLS_POSITIONS,
                                Nd=Nd, 
                                cutoff=0)      #  return correlations

elif Localization == 'GaspariCohn':
        locL = (8,8)
        locTheta = 45 * np.pi/180
        Rmd = BuildLocalizationMatrix(Ni, Nj, list_pos, locL, locTheta)
else:
    Rmd = np.eye(Nd)




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
alphas=np.ones(ESMDA_its)*ESMDA_its
MObj=np.zeros([len(alphas),Ne])
#Run



#%%
start= time.time()

for alpha in alphas:
    # Generates the perturbed observations (10.27)
    z = np.random.normal(size=(Nd, Ne))
    DPObs = dObs[:, np.newaxis] + math.sqrt(alpha) * CeDiag[:, np.newaxis] * z
    
    #prior_path = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/prior_geomodels'
    prior_path = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/hybrid_models/test1'

    destDir= os.path.join(curDir,f'{path_case}/it{l}')
    priorDir = os.path.join(curDir,f'{path_case}/it{0}')
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
                # filter non-physical values bellow zero
                Permeability[Permeability<0] = 0.01                
                #apply the log transform to the permeability
                # ----->>>>> Permeability = np.log(Permeability)
                Permeability = np.log(Permeability)
                
                                
                #Porosity = [] #realization['Por'].values.flatten()
                MGridPrior[:,i] = Permeability #np.concatenate((Permeability, Porosity))
                #save MGridPrior to pickle
                pd.DataFrame(MGridPrior).to_pickle(f'{destDir}/MGrid_{l}.pkl')
                MGrid = MGridPrior
                
        if RunPrior: 
            if NeHf > 0:         
                run_forward(reference_folder,
                    data_folder = geoDir,
                    numberHFmembers =NeHf,
                    Ne = Ne,
                    output_folder=dynDir,
                    is_proxy = False,
                    )
                #runs proxy
            if NePx > 0:
                run_forward(reference_folder,
                            data_folder = geoDir,
                            numberHFmembers=NeHf,
                            Ne = Ne,
                            output_folder=dynDir,
                            is_proxy = True,
                            )
            
        else:
            dynDir = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/prior_dynmodels/ensemble_100'    

    
        
    else:
    #read MGrid from the previous iteration
        destDir= os.path.join(curDir,f'{path_case}/it{l-1}')
        MGrid = pd.read_pickle(f'{destDir}/MGrid_{l-1}.pkl') 
        MGrid = MGrid.values
        destDir= os.path.join(curDir,f'{path_case}/it{l}')  
        geoDir= os.path.join(destDir,f'geo')
        dynDir= os.path.join(destDir,f'dyn')    
        for i, file in enumerate(os.listdir(prior_path)):
            if i < Ne:
                print(i)
                Permeability = MGrid[:,i]#MGrid[:int(NGrid/2),i]
                #Porosity = MGrid[int(NGrid/2):,i]                
                # ----->>>>>> Permeability = np.exp(Permeability)
                Permeability = np.exp(Permeability)
                Permeability = Permeability.reshape((Ni,Nj),order='F')
                Permeability[Permeability<0.01] = 0.01
                
                #Porosity = Porosity.reshape((Ni,Nj),order='F')
                #overwrite the values of the permeability and porosity in the realization
                realization = xr.open_dataset(os.path.join(prior_path,file))
                realization['Perm'].values = Permeability
                #realization['Por'].values = Porosity
                realization.to_netcdf(f'{geoDir}/geo_{i}.nc')

                print('saved geomodel to netcdf')
            
        print('Running...')

        if NeHf > 0:
        #runs DARTS
            run_forward(reference_folder,
                data_folder = geoDir,
                numberHFmembers =NeHf,
                Ne = Ne,
                output_folder=dynDir,
                is_proxy = False,
                )
        #runs proxy
        if NePx > 0:
            run_forward(reference_folder,
                        data_folder = geoDir,
                        numberHFmembers=NeHf,
                        Ne = Ne,
                        output_folder=dynDir,
                        is_proxy = True,
                        )
    
    D_full = compute_observation_matrix(dynDir, Ne, monitoring_positions, variable='Pressure',last_data_for_HM= 0)
    pd.DataFrame(D_full).to_pickle((f'{destDir}/D_full_{l}.pkl')) 
    
    D = compute_observation_matrix(dynDir, Ne, monitoring_positions, variable='Pressure', last_data_for_HM= last_data_for_HM)
    D_path= (f'{destDir}/D_{l}.pkl')
    Dprior_path= (f'{priorDir}/D_{0}.pkl')
    pd.DataFrame(D).to_pickle(D_path)   
    #comparer.process_data(prior_filename= Dprior_path, posterior_filename= D_path, figure_path= destDir,  num_realizations=100) 
    DobsD = DPObs - D
    
    meanMatrix = np.mean(D, axis=1)
    
    DeltaD = D - meanMatrix[:, np.newaxis]

    CHat = SInvDiag[:, np.newaxis] * \
        (DeltaD @ DeltaD.T) * \
        SInvDiag[np.newaxis, :] + alpha * (Ne - 1) * INd

    U, SigmaDiag, Vt = np.linalg.svd(CHat)
    csi = 0.99
    Nr = FindTruncationNumber(SigmaDiag, csi)

    GammaDiag = np.power(SigmaDiag[0:Nr], -1)
    X = SInvDiag[:, np.newaxis] * U[:, 0:Nr]   
    X1 = GammaDiag[:, np.newaxis] * X.T
    X8 = DeltaD.T @ X
    X9 = X8 @ X1
    
       

    if Localization is not None:
        MGrid = UpdateModelLocalized(MGrid, X9, Rmd, DobsD)
        
        #clip values bellow zero
        #MGrid[MGrid<0] = 0.01
        
        DeltaM = CentralizeMatrix(MGrid)
        K = DeltaM @ X9  
        K = Rmd * K
          
    else:
        MGrid = UpdateModel(MGrid, X9, DobsD)
        
        #clip values bellow zero
        #MGrid[MGrid<0] = 0.01
        
        DeltaM = CentralizeMatrix(MGrid)
        K = DeltaM @ X9
            
    #store K in a pickle file
    K_path= (f'{destDir}/K_{l}.pkl')   
    #save K to pickle
    pd.DataFrame(K).to_pickle(K_path)  
    
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
