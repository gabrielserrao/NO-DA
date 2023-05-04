import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
import os
import xarray as xr

#################################################
#
# Custom Dataset Class
#
#################################################

# Define the XarrayDataset class which inherits from Dataset class
class XarrayDataset(Dataset):
    # Initialize the class with folder, input_vars, and output_vars as input arguments
    def __init__(self, folder, input_vars, output_vars):
        self.folder = folder  # Set the folder containing the data files
        self.file_list = os.listdir(folder)  # List all the files in the folder
        self.input_vars = input_vars  # Set the input variables for the dataset
        self.input_vars.append('x_encoding')  # Add x_encoding to the input variables
        self.input_vars.append('y_encoding')  # Add y_encoding to the input variables
        self.input_vars.append('time_encoding')  # Add time_encoding to the input variables
        self.output_vars = output_vars  # Set the output variables for the dataset

    # Define the length method to return the number of files in the folder
    def __len__(self):
        return len(self.file_list)

    # Define the getitem method to return the input and output data for a specific index
    def __getitem__(self, idx):
        file_path = os.path.join(self.folder, self.file_list[idx])  # Get the file path for the given index
        data = xr.open_dataset(file_path)  # Open the file as an xarray dataset

        # Extract the X, Y, and time values from the dataset
        X = data['X'].values
        Y = data['Y'].values
        TIME = data['time'].values

        # Create meshgrids for X, Y, and time dimensions
        x_mesh, y_mesh = np.meshgrid(data.X, data.Y, indexing='ij')
        TIME_MESH = np.meshgrid(data.time, data.X, data.Y, indexing='ij')

        # Add the meshgrids as new data variables in the xarray dataset
        data = data.assign(x_encoding=xr.DataArray(x_mesh, coords=[("X", X), ("Y", Y)]))
        data = data.assign(y_encoding=xr.DataArray(y_mesh, coords=[("X", X), ("Y", Y)]))
        data = data.assign(time_encoding=xr.DataArray(TIME_MESH[0], coords=[("time", TIME), ("X", X), ("Y", Y)]))

        # Initialize an empty list to store the input data tensors
        input_data_list = []
        for var in self.input_vars:
            # Check if the variable has a time dimension
            if 'time' in data[var].dims:
                input_data_list.append(torch.tensor(data[var].values, dtype=torch.float32).unsqueeze(0))
            else:
                scalar_matrix = torch.tensor(data[var].values, dtype=torch.float32).expand(data.time.size, -1, -1)
                input_data_list.append(scalar_matrix.unsqueeze(0))

        # Concatenate the input data tensors along a new dimension
        input_data = torch.cat(input_data_list, dim=0)
        # Reorder dimensions to (time, shape[0], shape[1], channels)
        input_data = input_data.permute(1, 2, 3, 0)

        # Initialize an empty list to store the output data tensors
        output_data_list = [torch.tensor(data[var].values, dtype=torch.float32).unsqueeze(0) for var in self.output_vars]
        
        # Concatenate the output data tensors along a new dimension
        output_data = torch.cat(output_data_list, dim=0)
        # Reorder dimensions to (time, shape[0], shape[1], channels)
        output_data = output_data.permute(1, 2, 3, 0)  
        #final dimensions are (Batchsize, time, x, y, channels) 
        # Number of chanbels will be input variables + x_encoding, y_encoding, time_encoding (input + 3)
        return input_data, output_data