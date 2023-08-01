#%%
import sys
import os
import pickle
import numpy as np
import xarray as xr
import multiprocessing as mp

sys.path.append("./DARTS")
sys.path.append("./FNO3D")

from FNO3D.runProxy import run_proxy
from DARTS.runDARTS import run_DARTS_simulation

def run_DARTS_simulation_parallel(realization_path, i, treatGeoModel, dt, nsteps, well_coords, well_rates, initial_gas_rate, output_filename):
    realization = xr.open_dataset(realization_path)
    run_DARTS_simulation(realization,
                         treatGeoModel,
                         dt, 
                         nsteps,
                         well_coords,
                         well_rates,
                         initial_gas_rate,
                         output_filename=output_filename)


def run_forward(reference_folder,
                data_folder,
                numberHFmembers,
                Ne,
                output_folder,
                is_proxy = False):  

    with open(os.path.join(reference_folder,'Reference_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    dt = metadata.iloc[0].values[0]
    nsteps = metadata.iloc[1].values[0]
    well_coords = metadata.iloc[2].values[0]
    well_rates = metadata.iloc[3].values[0]
    initial_gas_rate = metadata.iloc[4].values[0]
    treatGeoModel = metadata.iloc[7].values[0]
    RefGeoData_path = metadata.iloc[8].values[0]

    if is_proxy:
        run_proxy(data_folder, 
                  path_model = '/samoa/data/smrserraoseabr/NO-DA/runs/FNO_3d_MonthQgWellCenter_N800.0_ep100_m18_w128_b1_normPointGaussianNormalizer_INPUT_Por_Perm_gas_rate_OUTPUT_Pressure/FNO_3d_MonthQgWellCenter_N800.0_ep100_m18_w128_b1_normPointGaussianNormalizer_INPUT_Por_Perm_gas_rate_OUTPUT_Pressure_model.pt',
                  input_vars = ['Por', 'Perm', 'gas_rate'],
                  output_vars = ['Pressure'],
                  WELLS_POSITIONS = True,
                  device = 'cpu',
                  output_folder = output_folder,
                  Ne=Ne)
    else:
        with mp.Pool(mp.cpu_count()) as pool:
            realizations_paths = [os.path.join(data_folder, realization) for realization in os.listdir(data_folder)[:Ne]]
            parameters_for_runs = []

            for i, realization_path in enumerate(realizations_paths):
                if i < numberHFmembers:
                    parameters_for_runs.append((realization_path, 
                                                i, 
                                                treatGeoModel, 
                                                dt, 
                                                nsteps, 
                                                well_coords, 
                                                well_rates, 
                                                initial_gas_rate, 
                                                f'{output_folder}/darts_out_{i}.nc'))

            pool.starmap(run_DARTS_simulation_parallel, parameters_for_runs)


# Specify the directories and parameters for your test
reference_folder = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/REFERENCE'
data_folder = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/prior_geomodels'
numberHFmembers = 10  # or your desired number
Ne = 20  # or your desired number
output_folder = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/teste_parallel'
is_proxy = False

# Ensure that the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Call the function with your specified parameters
run_forward(reference_folder, data_folder, numberHFmembers, Ne, output_folder, is_proxy)

# %%
