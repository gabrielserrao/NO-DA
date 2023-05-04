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
    
#################################################
#
# Load Data Class
#
#################################################
class ReadXarrayDataset():
    def __init__(self, folder, input_vars, output_vars, device='cpu', num_files=None, traintest_split= 0.8):
        self.folder = folder        
        self.file_list = os.listdir(folder)[:num_files] if num_files else os.listdir(folder)            
        self.input_vars = input_vars.copy()
        self.input_vars.append('x_encoding')
        self.input_vars.append('y_encoding')
        self.input_vars.append('time_encoding')
        self.output_vars = output_vars
        self.device = device
        
        input_data_list = []
        output_data_list = []
        
        #iterate inside the folder
        for i in range(len(self.file_list)):
            file_path = os.path.join(self.folder, self.file_list[i])
            data = xr.open_dataset(file_path)
            
            # Add the meshgrids as new data variables in the xarray
            X = data['X'].values
            Y = data['Y'].values
            TIME = data['time'].values

            # Create meshgrids for X and Y dimensions
            x_mesh, y_mesh = np.meshgrid(data.X, data.Y, indexing='ij')
            TIME_MESH = np.meshgrid(data.time, data.X, data.Y, indexing='ij')
            data = data.assign(x_encoding=xr.DataArray(x_mesh, coords=[("X", X), ("Y", Y)]))
            data = data.assign(y_encoding=xr.DataArray(y_mesh, coords=[("X", X), ("Y", Y)]))
            data = data.assign(time_encoding=xr.DataArray(TIME_MESH[0], coords=[("time", TIME), ("X", X), ("Y", Y)]))

            # Append input variables to list
            input_data = []
            for var in self.input_vars:
                if 'time' in data[var].dims:
                    input_data.append(torch.tensor(data[var].values, dtype=torch.float32))
                else:
                    scalar_matrix = torch.tensor(data[var].values, dtype=torch.float32).expand(data.time.size, -1, -1)
                    input_data.append(scalar_matrix)

            # Append output variables to list
            output_data = []
            for var in self.output_vars:
                output_data.append(torch.tensor(data[var].values, dtype=torch.float32))

            # Concatenate input variables along new dimension
            input_data_list.append(torch.stack(input_data, dim=-1))

            # Concatenate output variables along new dimension
            output_data_list.append(torch.stack(output_data, dim=-1))

        # Concatenate input variables along new dimension
        input_data = torch.stack(input_data_list, dim=0)

        # Concatenate output variables along new dimension
        output_data = torch.stack(output_data_list, dim=0)

        # Swap order of time and channel dimensions
        input_data = input_data.permute(0,1, 2, 3, 4)
        output_data = output_data.permute(0,1, 2, 3, 4)

        self.input_data = input_data
        self.output_data = output_data

        # Split data into training and testing sets
        self.train_size = int(traintest_split * len(self.file_list))
        self.test_size = len(self.file_list) - self.train_size
        
        self.train_data_input, self.test_data_input = torch.split(self.input_data, [self.train_size, self.test_size], dim=0)
        self.train_data_output, self.test_data_output = torch.split(self.output_data, [self.train_size, self.test_size], dim=0)

#################################################
#
# Normalization Class for Data
#
#################################################
# 
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = torch.mean(x, dim=(0, 1, 2))
        self.std = torch.std(x, dim=(0, 1, 2))
        self.eps = eps


    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        std = self.std + self.eps # n
        mean = self.mean
        x = (x * std) + mean
        return x
