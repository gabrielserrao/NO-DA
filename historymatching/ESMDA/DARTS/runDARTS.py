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
# %%

def run_DARTS_simulation(realization,
                   treatGeoModel=False,
                    dt=1, 
                    nsteps=100,
                    well_coords=[(2,2), (18,2) ,(2,18), (18,18)],
                    well_rates=[100, 100, 100, 100],
                    initial_gas_rate=0.0,
                    output_filename='RESULT.nc'):    
    start_time = 0  # in days
    end_time = nsteps * dt  
    times = dt * np.arange(nsteps+1)
    well_names = ['INJ1']#,'INJ2']
    well_types = ['INJECTOR']#, 'INJECTOR']

    if treatGeoModel == True:
        factor = 1.01324997e12
        RefGeoData['Perm'] = RefGeoData['Perm'] * factor
        # Set the minimum value of Perm to 0.001 mD
        RefGeoData['Perm'] = RefGeoData['Perm'].where(RefGeoData['Perm'] >= 0.001, 0.001)

        #Set the minimum value of Por to 0.01
        RefGeoData['Por'] = RefGeoData['Por'].where(RefGeoData['Por'] >= 0.01, 0.01)

   
    wells_config = create_wells_dataset(well_names, well_rates, well_types, well_coords, nsteps, initial_gas_rate)
    
    data = InitializeDataVars(realization, nsteps, times)
   
    m = Model(perm = realization['Perm'], por = realization['Por'], wells_config= wells_config, n_points=1000)
    m.init()
    m.run_python(1)
    data = StoreSimValues(m=m, size =32, data=data, t=0)

    for t in range(nsteps):
        m.set_wells(step = t)
        m.run_python(dt, restart_dt=1e-8)     
        data = StoreSimValues(m=m, size =32, data=data, t=t+1)
    m.print_timers()
    m.print_stat()

    wellsdata = ModelOut(m)
    wellsdata['new_time'] = wellsdata['time'] - 1
    wellsdata = wellsdata.where(wellsdata['new_time'].isin(times), drop=True)
    wellsdata = wellsdata.drop('new_time')
    wellsdata['time'] = wellsdata['time'] - 1
    wellsdata['time'] = wellsdata['time'].astype(int)
    data['time'] = data['time'].astype(int)
    combined_data = xr.merge([wellsdata, data])
    combined_data= combined_data.astype('float32')
    combined_data['i'] = (('origin'), wells_config['i'].values)
    combined_data['j'] = (('origin'), wells_config['j'].values)

    # Save the data with the provided output_filename
    combined_data.to_netcdf(output_filename)
    print('Job finished!')
    print(f'Results of DARTS saved to {output_filename}')






# %%
