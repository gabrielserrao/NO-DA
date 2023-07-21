#%%
import sys
sys.path.append("../..")
from darts.engines import redirect_darts_output
from model_co2 import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D
from utilsDARTS import ModelOut, InitializeDataVars, StoreSimValues,create_wells_dataset
import os
from runDARTS import run_DARTS_simulation
# %%
RefGeoData_path = '/samoa/data/smrserraoseabr/NO-DA/dataset/mixedcontext32x32/simulation_results_realization_32x32_1000.nc'
RefGeoData = xr.open_dataset(RefGeoData_path)
treatGeoModel =False
dt = 30
nsteps = 60
well_coords = [(16, 16)]
initial_gas_rate = 1000 

initial_gas_rate = initial_gas_rate * (1 + 0.5 * np.random.randn())  
well_rates = initial_gas_rate * np.exp(-np.linspace(0, 1, nsteps))

output_folder='.'
output_filename = os.path.join(output_folder,f'../REFERENCE/ReferenceSimulation.nc')
combined_data = run_DARTS_simulation(realization = RefGeoData,
                                    treatGeoModel = treatGeoModel,
                                    dt=dt,
                                    nsteps=nsteps, 
                                    well_coords = well_coords, 
                                    well_rates = well_rates,
                                    initial_gas_rate=initial_gas_rate, 
                                    output_filename=output_filename)

#create pkl file with associated input data
#%%
metadata = {'dt': dt,
            'nsteps': nsteps,
            'well_coords': well_coords,
            'well_rates': well_rates,
            'initial_gas_rate': initial_gas_rate, 
            'output_folder': output_folder,
            'output_filename': output_filename,
            'treatGeoModel': treatGeoModel,
            'RefGeoData_path': RefGeoData_path}


metadata_df = pd.DataFrame.from_dict(metadata, orient='index')
metadata_df.to_pickle(os.path.join(output_folder,f'../REFERENCE/Reference_metadata.pkl'))



# %%
