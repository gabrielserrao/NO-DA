#%%
import sys
sys.path.append("./DARTS")
sys.path.append("./FNO3D")
sys.path.append("../..")
from FNO3D.runProxy import run_proxy
from DARTS.runDARTS import run_DARTS_simulation
import pickle
import os
import numpy as np
import xarray as xr
from multiprocessing import Pool
from functools import partial
#%%
def run_simulation_for_realization(i, realization_file, treatGeoModel, dt, nsteps, well_coords, well_rates, initial_gas_rate, output_folder, numberHFmembers, data_folder):
    if i < numberHFmembers:
        realization_path = os.path.join(data_folder, realization_file)
        realization = xr.open_dataset(realization_path)
        run_DARTS_simulation(realization,
                             treatGeoModel,
                             dt, 
                             nsteps,
                             well_coords,
                             well_rates,
                             initial_gas_rate,
                             output_filename=f'{output_folder}/darts_out_{i}.nc')



def run_forward(reference_folder,
                data_folder,
                numberHFmembers,
                Ne,
                output_folder,
                is_proxy = False,
                ):  
    #reference_folder = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/REFERENCE'
    #read reference metadata from reference_folder

    with open(os.path.join(reference_folder,'Reference_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    dt = metadata.iloc[0].values[0]
    nsteps = metadata.iloc[1].values[0]
    well_coords = metadata.iloc[2].values[0]
    well_rates = metadata.iloc[3].values[0]
    initial_gas_rate = metadata.iloc[4].values[0]
    # output_folder = metadata.iloc[5].values[0]
    # output_filename = metadata.iloc[6].values[0]
    treatGeoModel = metadata.iloc[7].values[0]
    RefGeoData_path = metadata.iloc[8].values[0]

    if is_proxy:
    #chop the data_folder in Ne 
        run_proxy(data_folder, 
                path_model = \
                    '/samoa/data/smrserraoseabr/NO-DA/runs/FNO_3d_MonthQgWellCenter_N800.0_ep100_m18_w128_b1_normPointGaussianNormalizer_INPUT_Por_Perm_gas_rate_OUTPUT_Pressure/FNO_3d_MonthQgWellCenter_N800.0_ep100_m18_w128_b1_normPointGaussianNormalizer_INPUT_Por_Perm_gas_rate_OUTPUT_Pressure_model.pt',
                input_vars = ['Por', 'Perm', 'gas_rate'],
                output_vars = ['Pressure'],
                WELLS_POSITIONS = True,
                device = 'cpu',  # Use GPU
                output_folder = output_folder,
                Ne=Ne)

    else:
        realization_files = os.listdir(data_folder)[:Ne]
        run_simulation_partial = partial(run_simulation_for_realization,
                                         treatGeoModel=treatGeoModel,
                                         dt=dt,
                                         nsteps=nsteps,
                                         well_coords=well_coords,
                                         well_rates=well_rates,
                                         initial_gas_rate=initial_gas_rate,
                                         output_folder=output_folder,
                                         numberHFmembers=numberHFmembers,
                                         data_folder=data_folder)

        print("Starting simulations...")
        with Pool() as pool:
            pool.starmap(run_simulation_partial, enumerate(realization_files))
            pool.close()
            pool.join()
        print("Simulations finished.")
        # with Pool() as pool:
        #     results = [pool.apply_async(run_simulation_partial, args=(i, realization_file)) for i, realization_file in enumerate(realization_files)]
        # for result in results:
        #     result.get()


            









