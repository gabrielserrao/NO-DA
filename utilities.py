import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset, TensorDataset
import os
import xarray as xr

import scipy.io
import h5py


import operator
from functools import reduce
from functools import partial
import psutil

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f'Memory usage: {round(process.memory_info().rss / 1024 ** 3)} GB')

# Call this function at various points in your code to print memory usage

#################################################
#
# Custom Dataset Class
#
#################################################

class ReadXarrayDataset(Dataset):
    def __init__(self, folder, input_vars, output_vars, num_files=None, wells_positions=False):
        self.folder = folder        
        self.file_list = os.listdir(folder)[:num_files] if num_files else os.listdir(folder)            
        self.input_vars = input_vars.copy()
        self.input_vars.append('x_encoding')
        self.input_vars.append('y_encoding')
        self.input_vars.append('time_encoding')
        self.output_vars = output_vars.copy()
        self.wells_positions = wells_positions
        # self.input_normalizers = []
        # self.output_normalizers = []
        # self.file_stats = {}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder, self.file_list[idx])
        data = xr.open_dataset(file_path)

        X = data['X'].values
        Y = data['Y'].values
        TIME = data['time'].values

        x_mesh, y_mesh = np.meshgrid(data.X, data.Y, indexing='ij')
        TIME_MESH = np.meshgrid(data.time, data.X, data.Y, indexing='ij')
        data = data.assign(x_encoding=xr.DataArray(x_mesh, coords=[("X", X), ("Y", Y)]))
        data = data.assign(y_encoding=xr.DataArray(y_mesh, coords=[("X", X), ("Y", Y)]))
        data = data.assign(time_encoding=xr.DataArray(TIME_MESH[0], coords=[("time", TIME), ("X", X), ("Y", Y)]))
        
        input_data = []
        for var in self.input_vars:
            if 'time' in data[var].dims and 'X' in data[var].dims and 'Y' in data[var].dims:
                input_data.append(torch.tensor(data[var].values, dtype=torch.float32))
            elif 'X' in data[var].dims and 'Y' in data[var].dims:
                scalar_matrix = torch.tensor(data[var].values, dtype=torch.float32) 
                scalar_matrix = scalar_matrix.unsqueeze(0).expand(data.time.size, -1, -1)
                input_data.append(scalar_matrix)
            elif 'time' in data[var].dims:
                if self.wells_positions:
                    variable_matrix = torch.zeros((data.time.size, data.X.size, data.Y.size), dtype=torch.float32)
                    for origin in data.origin.values:
                        i = data.sel(origin=origin)['i'].values.item()
                        j = data.sel(origin=origin)['j'].values.item()
                        variable_matrix[:, i, j] = torch.tensor(data.sel(origin=origin)[var].values, dtype=torch.float32) 
                        input_data.append(variable_matrix)              
                else:
                    scalar_matrix = torch.tensor(data[var][0].values, dtype=torch.float32) 
                    scalar_matrix = scalar_matrix.unsqueeze(-1).unsqueeze(-1).expand(-1, data.X.size, data.Y.size)
                    input_data.append(scalar_matrix)


        output_data = []
        for var in self.output_vars:
           output_data.append(torch.tensor(data[var].values, dtype=torch.float32))


        input_data = torch.stack(input_data, dim=-1)
        output_data = torch.stack(output_data, dim=-1)
    
        return input_data, output_data  




    
#################################################
#
# Load Data Class
#
#################################################
class ReadXarray():
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
                if 'time' in data[var].dims and 'X' in data[var].dims and 'Y' in data[var].dims:
                    input_data.append(torch.tensor(data[var].values, dtype=torch.float32))
                elif 'X' in data[var].dims and 'Y' in data[var].dims:
                    scalar_matrix = torch.tensor(data[var].values, dtype=torch.float32) 
                    scalar_matrix = scalar_matrix.unsqueeze(0).expand(data.time.size, -1, -1)
                    input_data.append(scalar_matrix)
                elif 'time' in data[var].dims:
                    scalar_matrix = torch.tensor(data[var][0].values, dtype=torch.float32) #using [0] as I am geting info from the first well 
                    scalar_matrix = scalar_matrix.unsqueeze(-1).unsqueeze(-1).expand(-1, data.X.size, data.Y.size)
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
class GaussianNormalizer(object):
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
    
class UnitGaussianNormalizer(object):
    def __init__(self, x, mean=None, std=None, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()
        
        # If mean and std are provided, use them. Otherwise, compute from data.
        if mean is None or std is None:
            self.mean = torch.mean(x, dim=(0, 1, 2))
            self.std = torch.std(x, dim=(0, 1, 2))
        else:
            self.mean = mean
            self.std = std
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        std = self.std + self.eps # n
        mean = self.mean
        x = (x * std) + mean
        return x
    
    def decode_with_values(self, x, mean, std):
        x = (x * std) + mean
        return x
    
# normalization, pointwise gaussian
class PointGaussianNormalizer(object):
    def __init__(self, dataloader, mean=None, std=None, is_label=False, eps=0.00001):
        super(PointGaussianNormalizer, self).__init__()
        self.eps = eps
        if mean is None or std is None:
            self.mean, self.std = self.batch_mean_and_sd(dataloader, is_label)
        else:
            self.mean = mean
            self.std = std

    def batch_mean_and_sd(self, loader, is_label):
        cnt = 0
        fst_moment = None
        snd_moment = None

        for data, labels in loader:
            data = labels if is_label else data  # if this is label normalizer, normalize labels
            data_shape = data.shape[-1]
            if fst_moment is None:
                fst_moment = torch.zeros(data_shape)
                snd_moment = torch.zeros(data_shape)

            b, t, h, w, c = data.shape
            nb_pixels = b * t * h * w
            sum_ = torch.sum(data, dim=[0, 1, 2, 3])
            sum_of_square = torch.sum(data ** 2, dim=[0, 1, 2, 3])
            fst_moment += sum_
            snd_moment += sum_of_square
            cnt += nb_pixels

        fst_moment /= cnt
        snd_moment /= cnt

        return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

    def cuda(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self 



#################################################
#
# Custom Loss Function
#
#################################################
#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss

# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c


