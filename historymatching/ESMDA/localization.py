import sys
import os
import numpy as np
import pandas as pd
import torch
from utilities import *
from model_fourier_3d import *
from scipy.stats import pearsonr

def kproxy_corr_localization2(data_folder,
                            path_model, 
                            input_vars, 
                            output_vars, 
                            device, 
                            output_folder, 
                            Ne, 
                            monitoring_positions,
                            WELLS_POSITIONS,
                            Nd, 
                            cutoff):
    sys.path.append("..")

    print(torch.__version__)
    print(f"GPUs:{torch.cuda.device_count()}")
    print(os.getcwd())
  
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

    model = torch.load(path_model, map_location=device)
    model.to(device)
    model.eval()

    global_count = 0  # Global file counter

    pressure_ensemble = []
    perm_ensemble = []

    for i, (x, _) in enumerate(data_loader):
        Y = x.shape[1]
        X = x.shape[2]
        perm = x[:, 0, :, :, 1].detach().cpu().numpy().flatten()  # Adjust this if the permeability is not at index 1
        x = x.to(device)
        x = input_normalizer.encode(x)
        out = model(x)
        pressure = output_normalizer.decode(out).detach().cpu().numpy()  
        
        #for each monitoring point compute the pressure for all steps  
        # Get the pressure at the specific grid point and the time step 
        
        for (i,j) in monitoring_positions:
            pressure_ensemble.append(pressure[:, :, i, j, 0])

        perm_ensemble.append(perm)
    #%%
    perm_ensemble = np.array(perm_ensemble)
    pressure_ensemble = np.array(pressure_ensemble)
    #%%
    pressure_ensemble_reshaped = pressure_ensemble.reshape((len(monitoring_positions) * x.shape[1], Ne))
    perm_ensemble_transposed = perm_ensemble.T
    correlation_matrix = np.zeros((1024, 244))
    # Iterate through each permeability grid point and each monitoring data
    # and compute the correlation
    from scipy.stats import pearsonr
    for i in range(1024):
        for j in range(244):
            correlation_matrix[i, j], _ = pearsonr(perm_ensemble_transposed[i], pressure_ensemble_reshaped[j])
            # Set the value to 0 if the absolute value is below the cutoff
            if abs(correlation_matrix[i, j]) < cutoff:
                correlation_matrix[i, j] = 0
    
    # Save correlations to pickle
    pd.DataFrame(correlation_matrix).to_pickle(f'{output_folder}/correlation_matrix_Ne_{Ne}_cutoff_{cutoff}.pkl')
    
    #normalizing the correlation matrix by min-max
    min_val = np.min(correlation_matrix)
    max_val = np.max(correlation_matrix)
    localization_matrix = (correlation_matrix - min_val) / (max_val - min_val)
    
    pd.DataFrame(localization_matrix).to_pickle(f'{output_folder}/localization_matrix_Ne_{Ne}_cutoff_{cutoff}.pkl')
    
    return localization_matrix
    """
    Compute the localization matrix using Furrer and Bengtsson Taper (corrected version).

    :param correlation_matrix: Correlation matrix between the variables (e.g., ensemble estimates)
    :param Ne: Ensemble size
    :param eta: Threshold for zeroing small values
    :return: Localization matrix
    """
    # Number of variables
    # eta=0.05
    # n = correlation_matrix.shape[0]
    # k = correlation_matrix.shape[1]
    
    # # Compute the covariance matrix (in this case, correlation matrix itself can be used)
    # covariance_matrix = correlation_matrix

    # # Initialize the localization matrix
    # localization_matrix = np.zeros_like(correlation_matrix)

    # Compute the localization matrix using the given formula
    # for i in range(n):
    #     for j in range(k):
    #         c_ij = covariance_matrix[i, j]
    #         c_ii = covariance_matrix[i, i]
    #         c_jj = covariance_matrix[j, j]

    #         r_ij = (c_ij ** 2) / ((c_ij ** 2) + ((c_ij ** 2) + c_ii * c_jj) / Ne)

    #         # Apply the threshold
    #         threshold = eta * np.sqrt(c_ii * c_jj)
    #         if np.abs(c_ij) < threshold:
    #             r_ij = 0

    #         localization_matrix[i, j] = r_ij


    