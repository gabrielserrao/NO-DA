import os
import time 
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path

from runForward import run_forward
from utilsESMDA import (ModelOut, deleteOutFiles, media_function, get_grid_outputs,
                        CalcHL, SphereFunction, GaspariCohnFunction, IndexToIJ, 
                        IJToIndex, BuildPermCovMatrix, BuildLocalizationMatrix,
                        PlotModelRealization, PlotMatrix, RunModels, check_job,
                        ReadModels, FindTruncationNumber, CentralizeMatrix, 
                        UpdateModelLocalized, UpdateModel, calcDataMismatchObjectiveFunction, 
                        MultipliNegatives)

# Constants
REFERENCE_FOLDER = Path('/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/REFERENCE')
MONITORING_POSITIONS = [[2,2], [18,2] ,[2,18], [18,18]] 
PRIOR_PATH = Path('/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/prior_geomodels')

def load_reference_metadata(reference_folder):
    with open(reference_folder / 'Reference_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return metadata

def compute_observation_matrix(directory_path, Ne, monitoring_positions):
    D = []
    for i in range(Ne):
        file_path = directory_path / (f"darts_out_{i}.nc" if (directory_path / f"darts_out_{i}.nc").exists() else f"proxy_out_{i}.nc")
        model = xr.open_dataset(file_path)
        obsData = [model['Pressure'].isel(X=x, Y=y).values for x, y in monitoring_positions]
        D.append(np.array(obsData).flatten())
    D = np.array(D).T
    return D

def run_iteration(Ne, NeHf, monitoring_positions, reference_folder, iteration_number, alphas):
    destDir = Path.cwd() / f'simulations/it{iteration_number}'
    geoDir = Path.cwd() / f'simulations/geo'
    dynDir = destDir / 'dyn' 
    destDir.mkdir(parents=True, exist_ok=True)
    geoDir.mkdir(parents=True, exist_ok=True)
    dynDir.mkdir(parents=True, exist_ok=True)

    MGridPrior = np.empty([Nm, Ne]) if iteration_number == 0 else pd.read_pickle(destDir / f'MGrid_{iteration_number - 1}.pkl')
    
    for i, file in enumerate(PRIOR_PATH.iterdir()):
        if i >= Ne:
            break

        realization = xr.open_dataset(file)
        realization.to_netcdf(geoDir / f'geo_{i}.nc')

        if iteration_number == 0:
            Permeability = np.log(realization['Perm'].values.flatten())
            Porosity = realization['Por'].values.flatten()
            MGridPrior[:,i] = np.concatenate((Permeability, Porosity))
            pd.DataFrame(MGridPrior).to_pickle(destDir / f'MGrid_{iteration_number}.pkl')
        else:
            Permeability = np.exp(MGridPrior[:int(NGrid/2),i])
            Permeability = Permeability.reshape((Ni, Nj), order='F')
            Porosity = MGridPrior[int(NGrid/2):,i]
            Porosity = Porosity.reshape((Ni, Nj), order='F')
            realization['Perm'].values = Permeability
            realization['Por'].values = Porosity
            realization.to_netcdf(geoDir / f'geo_{i}.nc')

    if iteration_number != 0:
        run_forward(reference_folder, geoDir, NeHf, Ne, dynDir, is_proxy=False)
        run_forward(reference_folder, geoDir, NeHf, Ne, dynDir, is_proxy=True)

    D = compute_observation_matrix(dynDir, Ne, monitoring_positions)
    pd.DataFrame(D).to_pickle(destDir / f'D_{iteration_number}.pkl')
    return MGridPrior, D

def data_assimilation(Ne, NeHf, monitoring_positions, reference_folder, iterations):
    alphas = np.ones(iterations) * iterations
    MObj = np.zeros([len(alphas), Ne])
    MObjMean = np.zeros(iterations)

    for l in range(iterations):
        MGridPrior, D = run_iteration(Ne, NeHf, monitoring_positions, reference_folder, l, alphas)
        # Rest of the operations for each iteration...

    return MGridPrior, D, MObj

#def main():
metadata = load_reference_metadata(REFERENCE_FOLDER)
Ne = metadata.iloc[7].values[0]
NeHf = metadata.iloc[1].values[0]

start = time.time()
MGridPost, DPost, MObj = data_assimilation(Ne, NeHf, MONITORING_POSITIONS, REFERENCE_FOLDER, 2)
print('Elapsed time of the ES-MDA: ', time.time() - start)

# if __name__ == "__main__":
#     main()
