# %%
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from utilities import *
import imageio
from io import BytesIO
from IPython.display import Image as DisplayImage
from model_fourier_3d import *
import torchmetrics
print(torch.__version__)
print(f"GPUs:{torch.cuda.device_count()}")
import os
print(os.getcwd())

# %%
###############################################
#INTIAL CONFIGS
# DATASET
FOLDER = "../dataset/mixedcontext32x32"  #"/nethome/atena_projetos/bgy3/NO-DA/datasets/results" + str(resolution) + "/"
INPUT_VARS = ['Por', 'Perm', 'gas_rate'] # Porosity, Permeability, ,  Well 'gas_rate', Pressure + x, y, time encodings 
OUTPUT_VARS = ['Pressure'] 

#CONFIGS OF THE MODEL TO GENERATE RESULTS
BASE_PATH = '/samoa/data/smrserraoseabr/NO-DA/runs'

NUM_FILES= 1000
TRAINTEST_SPLIT = 0.8
BATCH_SIZE = 10
EPOCHS = 110
MODES = 18
WIDTH = 128

#List of samples to plot:
BATCH_TO_PLOT = [0]
SAMPLES_TO_PLOT = [0, 1, 2, 3, 4, 5, 6, 7, 8]

#DEVICE SETTINGS
device = 'cpu'
#OUTPUT CONFIGURATION
EVALUATE_METRICS = True
plot_model_eval = True
plot_comparison = True
plot_lines = True
plot_gifs =True
###############################################
variable = OUTPUT_VARS[0]
ntrain = NUM_FILES * TRAINTEST_SPLIT
ntest = NUM_FILES - ntrain
path = 'FNO_3d_N{}_ep{}_m{}_w{}_b{}'.format(ntrain, EPOCHS, MODES, WIDTH, BATCH_SIZE)
path += '_INPUT_' + '_'.join(INPUT_VARS) + '_OUTPUT_' + '_'.join(OUTPUT_VARS)
path_runs = os.path.join(BASE_PATH, path)
path_model = os.path.join(path_runs, f'{path}_model.pt')
path_normalizer = path_runs
image_folder = os.path.join(path_runs, 'images')
log_folder = os.path.join(path_runs, 'log')

if not os.path.exists(image_folder):
    os.makedirs(image_folder)

#check  all paths
print(f'Path runs: {path_runs}')
print(f'Path model: {path_model}')

#CHCEK IF FOLDERS EXISTS
if not os.path.exists(path_runs):
    raise ValueError(f'Path {path_runs} does not exist.')
if not os.path.exists(path_model):
       raise ValueError(f'Path {path_model} does not exist.')
if not os.path.exists(image_folder):
    raise ValueError(f'Path {image_folder} does not exist.')


if variable == 'CO_2':
    colorbar_vmax, colorbar_vmin = 1.0, 0.0 # Define your min and max here
elif variable == 'Pressure':
    colorbar_vmin, colorbar_vmax = 200.0, 600.0 # Define your min and max here
  # Change this to the index you want
#%%     
###############################################
#LOAD DATA

# Create instance of ReadXarrayDatasetNorm class for training data
dataset = ReadXarrayDataset(folder=FOLDER, input_vars=INPUT_VARS, output_vars=OUTPUT_VARS, num_files = NUM_FILES, wells_positions=False)

train_size = int(TRAINTEST_SPLIT * len(dataset))
test_size = len(dataset) - train_size


train_loader = DataLoader(torch.utils.data.Subset(dataset, range(0, train_size)),
                           batch_size=BATCH_SIZE,
                             shuffle=False)
test_loader = DataLoader(torch.utils.data.Subset(dataset, range(train_size, train_size + test_size)), 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)
# We no longer have the entire dataset loaded into memory. The normalization is handled by the Dataset class.

input_normalizer = PointGaussianNormalizer(train_loader, is_label=False)
output_normalizer = PointGaussianNormalizer(train_loader, is_label=True)

# %%
#LOAD NORMALIZATION PARAMETERS

# Normalize input_data and output_data
#check if normalizer exists and load it, otherwise raise a warning and create a new one
if os.path.exists(os.path.join(path_runs,'normalizer_mean_input.pt')):
    input_normalizer_mean = torch.load(os.path.join(path_runs,'normalizer_mean_input.pt'))
    input_normalizer_std = torch.load(os.path.join(path_runs,'normalizer_std_input.pt'))
    output_normalizer_mean = torch.load(os.path.join(path_runs,'normalizer_mean_output.pt'))
    output_normalizer_std = torch.load(os.path.join(path_runs,'normalizer_std_output.pt'))
    print('Normalizer loaded')
    input_normalizer = PointGaussianNormalizer(train_loader, mean = input_normalizer_mean, std = input_normalizer_std, is_label=False)
    output_normalizer = PointGaussianNormalizer(train_loader, mean = output_normalizer_mean, std = output_normalizer_std, is_label=True)

    input_normalizer = input_normalizer.cuda(device)
    output_normalizer = output_normalizer.cuda(device)



else:
    print('Normalizer not found')
    #raise 
    ValueError(f'Normalizer not found in {path_runs}')

#%%
###############################################
device = torch.device("cpu")
model = torch.load(path_model, map_location=device).to(device)  # load the model to CPU
model.eval()  # set the model to evaluation mode
#extract main parameters of model and print them like padding, number of layers, etc
print('Model loaded')
#print number of parameters of model
print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
#%%
###############################################
#OVERALL MODEL EVALUATION
def extract_data(file_name):
    epoch_list = []
    mse_list = []
    l2_list = []

    with open(file_name, 'r') as file:
        for line in file:
            if 'epoch' in line:
                # Using regex to extract values
                epoch = re.search('epoch (.*): t', line).group(1)
                mse = re.search('mse=(.*),', line).group(1)
                l2 = re.search('l2=(.*)', line).group(1)
                epoch_list.append(int(epoch))
                mse_list.append(float(mse))
                l2_list.append(float(l2))
    return epoch_list, mse_list, l2_list

if plot_model_eval:   
   # Extract data
    epoch_train, mse_train, l2_train = extract_data(os.path.join(log_folder, 'train.txt'))

    epoch_test, mse_test, l2_test =  extract_data(os.path.join(log_folder, 'test.txt'))

    # Create figure and axis
    fig, ax1 = plt.subplots()

    # Plot MSE and L2 for both test and train datasets
    ax1.plot(epoch_train, l2_train, 'b-', label='Train L2')
    ax1.plot(epoch_test, l2_test, 'b-' , label='Test L2')

    # Set labels and title
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Value')
    ax1.set_title('Comparison of Train and Test L2 values')
    ax1.legend()
    plt.savefig(os.path.join(path_runs, f'{path_runs}_model_eval_L2.png'))

        # Create figure and axis
    fig, ax1 = plt.subplots()

    # Plot MSE and L2 for both test and train datasets
    ax1.plot(epoch_train, mse_train, 'g-', label='Train MSE')
    ax1.plot(epoch_test, mse_test, 'g-', label='Test MSE')

    # Set labels and title
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Value')
    ax1.set_title('Comparison of Train and Test MSE values')
    ax1.legend()
    plt.savefig(os.path.join(path_runs, f'{path_runs}_model_eval_MSE.png'))




#%%
###############################################
#GENERATE IMAGES AND PLOTS FOR EACH MODEL
def plot_to_memory_image(true, predicted, time_step, variable):
    buf = BytesIO()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.imshow(true, cmap='jet')
    ax2.imshow(predicted, cmap='jet')
    ax1.set_title(f'True {variable} at time step {time_step}')
    ax2.set_title(f'Predicted {variable} at time step {time_step}')
    for ax in (ax1, ax2):
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return buf

#%%
if EVALUATE_METRICS:
# Instantiate the metric
    metric = torchmetrics.MeanSquaredError()

    mse_scores = []
    std_parameter_samples = []

    # Iterate over test data
    for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            true_y = y.to(device)
            x = input_normalizer.encode(x)
            out = model(x)
            out = output_normalizer.decode(out)
            x = input_normalizer.decode(x)
            num_samples = x.size(0)
            for index in range(num_samples):
                sample = batch_idx * BATCH_SIZE + index
                test_y = true_y[index,...].detach().cpu()
                predicted_y = out[index,...].detach().cpu()
                 # Compute and store the Mean Squared Error
                mse = metric(output_normalizer.encode(predicted_y), output_normalizer.encode(test_y))
                print(f"Sample {sample} - MSE: {mse.item()}")
                mse_scores.append((sample, mse.item()))  # .item() is used to get a Python number from a tensor containing a single value
                #compute the std of the the permeability of the sample
                std = torch.std(x[index,0,:,:,1]) #permeability
                std_parameter_samples.append((sample, std.item()))

    # Create a plot
    indices, scores = zip(*mse_scores)  # Unpack mse_scores

    fig, ax = plt.subplots()
    #scatter plot
    ax.scatter(indices, scores)
    #mean line
    ax.axhline(np.mean(scores), color='red', linestyle='dashed', linewidth=1)
    ax.set_xlabel('Sample')
    ax.set_ylabel('MSE')
    ax.set_title('MSE of all models')
    plt.savefig(os.path.join(log_folder, 'mse_scores.jpg'))
    plt.close()

    #save a text file on logs folder with the MSE of all models 
    with open(os.path.join(log_folder, 'mse_scores.txt'), 'w') as file:
        for sample, mse in mse_scores:
            file.write(f'Sample {sample} - MSE: {mse}\n')

    #plot the MSE of the sample against the std of the permeability. A scatter plot of mse and std of the permeability
    indices, std_param = zip(*std_parameter_samples)  # Unpack mse_scores
    fig, ax = plt.subplots()
    ax.scatter(std_param, scores)
    ax.set_xlabel('Std of the permeability')
    ax.set_ylabel('MSE')
    ax.set_title('MSE of all models')
    plt.savefig(os.path.join(log_folder, 'mse_scores_std.jpg'))
    plt.close()




#%%   

#%%
for batch_idx, (x, y) in enumerate(test_loader):        
        if batch_idx in BATCH_TO_PLOT:
            #check if batch_idx is in samples to plot
            x = x.to(device)
            true_y = y.to(device)
            x = input_normalizer.encode(x)
            out = model(x)
            out = output_normalizer.decode(out)
            num_samples = x.size(0)
            
            if plot_comparison:
                for index in SAMPLES_TO_PLOT:
                    sample = batch_idx * BATCH_SIZE + index
                    test_y = true_y[index,...].detach().cpu()
                    predicted_y = out[index,...].detach().cpu()

                    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
                    norm = mpl.colors.Normalize(vmin=colorbar_vmin, vmax=colorbar_vmax)

                    img1 = axes[0].imshow(test_y[-1, :, :, 0], cmap='jet', norm=norm)
                    img2 = axes[1].imshow(predicted_y[-1, :, :, 0], cmap='jet', norm=norm)
                    img3 = axes[2].imshow(np.abs(test_y[ -1, :, :, 0] - predicted_y[ -1, :, :, 0]), cmap='jet')

                    for img, ax in zip([img1, img2, img3], axes):
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        fig.colorbar(img, cax=cax, orientation='vertical')

                    axes[0].set_title(f'Test - Sample {sample+1}')
                    axes[1].set_title(f'Predicted  - Sample {sample+1}')
                    axes[2].set_title(f'Absolute Error - Sample {sample+1}')

                    for ax in axes:
                        ax.axis('off')

                    fig.suptitle(f'Comparison of Test and Predicted {variable} values for 3D Fourier Neural Operator')
                    plt.tight_layout()
                    plt.savefig(os.path.join(image_folder, f"comparison_{sample+1}.png"))
                    print(f"Comparison_{sample+1}.png saved")
                    plt.close() 
            
                    if plot_lines:
                        resolution  = test_y.shape[1]
                        x_line = np.arange(0, resolution, 4)
                        y_line = np.arange(0, resolution, 4)
                        time = test_y[:,0,0,0]
                        diagonal = [(i, i) for i in range(len(x_line))]
                        x_mid = [(int(resolution/2), i) for i in range(len(x_line))]
                        y_mid = [(i, int(resolution/2)) for i in range(len(y_line))]

                        for line in [diagonal, x_mid, y_mid]:
                            for i, (x, y) in enumerate(line):
                                fig, main_ax = plt.subplots()

                                main_ax.plot(time, test_y[:, x, y, 0], label='True', linestyle='solid', color = 'blue')
                                main_ax.plot(time, predicted_y[:, x, y, 0], label='Predicted', linestyle='none', marker='o', color = 'red')
                                main_ax.legend()

                                main_ax.set_xlabel('Time')
                                main_ax.set_ylabel(variable)
                                main_ax.set_title(f'Sample {sample+1} {variable} at x= {x} and y = {y} ')
                                main_ax.set_ylim([colorbar_vmin, colorbar_vmax])

                                left, bottom, width, height = [0.15, 0.45, 0.3, 0.3] # adjust as needed
                                inset_ax = fig.add_axes([left, bottom, width, height])
                                im = inset_ax.imshow(test_y[-1, :, :, 0], cmap='jet')
                                inset_ax.scatter(x, y, s=20, edgecolor='black', facecolor='none', linewidth=2) 

                                inset_ax.axis('off')

                                fig.savefig(os.path.join(image_folder, f'Sample_{sample+1}_comparison_point_x{x}_y{y}.png'), dpi=300)
                                print(f'Sample_{sample+1}_comparison_point_x{x}_y{y}.png saved')
                                plt.close()

                    if plot_gifs:
                        gif_paths = []
                        image_buffers = []

                        for t in range(61): # Assuming you have 61 time steps
                            buf = plot_to_memory_image(test_y[t, :, :, 0], predicted_y[t, :, :, 0], t, variable = variable)
                            image_buffers.append(buf)

                        images = [imageio.imread(buf.getvalue()) for buf in image_buffers]
                        buf_result = BytesIO()
                        imageio.mimsave(buf_result, images, format='GIF', duration=0.5)

                        # Save the GIF
                        gif_save_path = os.path.join(image_folder, f'Comparison_{sample+1}.gif')
                        with open(gif_save_path, 'wb') as f:
                            f.write(buf_result.getvalue())
                            buf_result.seek(0)

                        for buf in image_buffers:
                            buf.close()
                        # Display the GIF (assuming you are in Jupyter notebook)
                        print(f'Comparison_{sample+1}.gif saved')

                        gif_paths.append(gif_save_path)

# %%
