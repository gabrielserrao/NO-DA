#%%
import torch
import os
from utilities import *
from model_fourier_3d import *
from torch.utils.data import DataLoader
import re
import sys
sys.path.append("..")
sys.path.append(".")

def run_proxy(data_folder, 
                path_model = \
                    '/samoa/data/smrserraoseabr/NO-DA/runs/FNO_3d_N800.0_ep110_m18_w128_b10_INPUT_Por_Perm_gas_rate_OUTPUT_Pressure/FNO_3d_N800.0_ep110_m18_w128_b10_INPUT_Por_Perm_gas_rate_OUTPUT_Pressure_model.pt',
                input_vars = ['Por', 'Perm', 'gas_rate'],
                output_vars = ['Pressure'],
                WELLS_POSITIONS = True,
                device = 'cuda',  # Use GPU
                output_folder = '.',
                Ne=100):
        
        folder = os.path.dirname(path_model)       
        batch_size = 1  # Use batch size of 1
        num_files = Ne

        dataset = ReadXarrayDataset(folder=data_folder, 
                                input_vars=input_vars, 
                                output_vars=output_vars,
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

        input_normalizer = input_normalizer.cuda(device)
        output_normalizer = output_normalizer.cuda(device)

        model = torch.load(path_model)
        model.to(device)
        model.eval()

        global_count = 0  # Global file counter

        for i, (x, _) in enumerate(data_loader):
            time = x.shape[0]
            Y = x.shape[1]
            X = x.shape[2]
            x = x.to(device)
            x = input_normalizer.encode(x)
            out = model(x)
            out = output_normalizer.decode(out)
            out = out.detach().cpu().numpy()  
            print(f'Proxy simulation {global_count} completed!')

            for o in out:       
                o = np.squeeze(o)
                data_array = xr.DataArray(o, dims=('time', 'Y', 'X'))
                data_array.name = 'Pressure' 
                data_array.to_netcdf(f'{output_folder}/proxy_out_{global_count}.nc')
                print(f'Proxy simulation {global_count} saved')  # Use global counter
                global_count += 1  # Increment global counter after each file

# %%
