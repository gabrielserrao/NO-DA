# %%
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
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
from torchinfo import summary
print(os.getcwd())

# %%
###############################################
#INTIAL CONFIGS
# DATASET
TAG = 'MonthQgWellRand'
FOLDER = '../dataset/DARTS/runnedmodels_wells/filtered' #'/samoa/data/smrserraoseabr/NO-DA/dataset/mixedcontext32x32' #"../dataset/DARTS/runnedmodels_wells/filtered"  #"/nethome/atena_projetos/bgy3/NO-DA/datasets/results" + str(resolution) + "/"
INPUT_VARS = ['Por', 'Perm', 'gas_rate'] # Porosity, Permeability, ,  Well 'gas_rate', Pressure + x, y, time encodings 
OUTPUT_VARS = ['Pressure'] 
#CONFIGS OF THE MODEL TO GENERATE RESULTS
BASE_PATH = '/samoa/data/smrserraoseabr/NO-DA/runs'
NUM_FILES= 1000
TRAINTEST_SPLIT = 0.8
BATCH_SIZE = 1
EPOCHS = 100
MODES = 18
WIDTH = 128
NORMALIZER = 'PointGaussianNormalizer'
WELLS_POSITIONS = True
#List of samples to plot:
BATCH_TO_PLOT = [0, 1, 2, 3, 4, 5, 6, 7, 8]
SAMPLES_TO_PLOT = [0]#, 1, 2, 3, 4, 5, 6, 7, 8]
#DEVICE SETTINGS
device = 'cpu'
#OUTPUT CONFIGURATION
EVALUATE_METRICS = False
plot_model_eval = True
plot_comparison = True
plot_lines =True
plot_gifs =True
###############################################
variable = OUTPUT_VARS[0]
ntrain = NUM_FILES * TRAINTEST_SPLIT
ntest = NUM_FILES - ntrain

path = 'FNO_3d_{}_N{}_ep{}_m{}_w{}_b{}_norm{}'.format(TAG,ntrain, EPOCHS, MODES, WIDTH, BATCH_SIZE, NORMALIZER)
path += '_INPUT_' + '_'.join(INPUT_VARS) + '_OUTPUT_' + '_'.join(OUTPUT_VARS)
#path = '/samoa/data/smrserraoseabr/NO-DA/runs/FNO_3d_N800.0_ep200_m18_w128_b10_INPUT_Por_Perm_gas_rate_OUTPUT_CO_2'
path_runs = os.path.join(BASE_PATH, path)
path_model = os.path.join(path_runs, f'{path}_model.pt')
#path_model='/samoa/data/smrserraoseabr/NO-DA/runs/FNO_3d_N800.0_ep200_m18_w128_b10_INPUT_Por_Perm_gas_rate_OUTPUT_CO_2/FNO_3d_N800.0_ep200_m18_w128_b10_INPUT_Por_Perm_gas_rate_OUTPUT_CO_2_model.pt'
path_normalizer = path_runs
image_folder = os.path.join(path_runs, 'images')
log_folder = os.path.join(path_runs, 'log')

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

if not os.path.exists(image_folder):
    os.makedirs(image_folder)


if variable == 'CO_2':
    colorbar_vmax, colorbar_vmin = 1.0, 0.0 # Define your min and max here
elif variable == 'Pressure':
    colorbar_vmin, colorbar_vmax = 200.0, 300.0 # Define your min and max here
  # Change this to the index you want
#%%     
###############################################
#LOAD DATA

# Create instance of ReadXarrayDatasetNorm class for training data
dataset = ReadXarrayDataset(folder=FOLDER, input_vars=INPUT_VARS, output_vars=OUTPUT_VARS, num_files = NUM_FILES, wells_positions=WELLS_POSITIONS)

train_size = int(TRAINTEST_SPLIT * len(dataset))
test_size = len(dataset) - train_size


train_loader = DataLoader(torch.utils.data.Subset(dataset, range(0, train_size)),
                           batch_size=BATCH_SIZE,
                             shuffle=False)
test_loader = DataLoader(torch.utils.data.Subset(dataset, range(train_size, train_size + test_size)), 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)
# We no longer have the entire dataset loaded into memory. The normalization is handled by the Dataset class.

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
    #create a new normalizer
    if NORMALIZER == 'PointGaussianNormalizerNoNaN':
        input_normalizer = PointGaussianNormalizerNoNaN(train_loader, mean = input_normalizer_mean, std = input_normalizer_std, is_label=False)
        output_normalizer = PointGaussianNormalizerNoNaN(train_loader, mean = output_normalizer_mean, std = output_normalizer_std, is_label=True)
    elif NORMALIZER == 'PointGaussianNormalizer':
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
summary(model, input_size=(1, 61, 32, 32, 6))
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
    plt.savefig(os.path.join(log_folder, f'model_eval_L2.png'))

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
    plt.savefig(os.path.join(log_folder, f'model_eval_MSE.png'))




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

    entropy_scores = []
    connectivity_scores = []
    threshold = 10  # adjust as needed
    neighborhood_radius = 10  # adjust as needed

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

                entropy_map, mean_entropy = compute_entropy(x[index,0,:,:,1], neighborhood_radius)
                binary_map, num_components = compute_connectivity(x[index,0,:,:,1], threshold)                
                entropy_scores.append((sample, mean_entropy))
                connectivity_scores.append((sample, num_components))

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

    # Plot MSE against Mean Entropy
    mse_indices, mse_scores = zip(*mse_scores)
    entropy_indices, entropy_scores = zip(*entropy_scores)
    fig, ax = plt.subplots()
    ax.scatter(mse_scores, entropy_scores)
    
    ax.set_xlabel('MSE')
    ax.set_ylabel('Entropy')
    ax.set_title('MSE vs Entropy of all models')
    plt.savefig(os.path.join(log_folder, 'mse_vs_entropy.jpg'))
    plt.close()

    # Plot MSE against Number of Connected Components
    connectivity_indices, connectivity_scores = zip(*connectivity_scores)
    fig, ax = plt.subplots()
    ax.scatter(mse_scores, connectivity_scores)
    ax.set_xlabel('MSE')
    ax.set_ylabel('Number of Connected Components')
    ax.set_title('MSE vs Connectivity of all models')
    plt.savefig(os.path.join(log_folder, 'mse_vs_connectivity.jpg'))
    plt.close()


#%%
predicted_distances = []
true_distances = []
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

                    data = predicted_y[-1, :, :, 0].numpy()
                    true_data = test_y[-1, :, :, 0].numpy()
                    
                    X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
                    flat_data  = data.flatten()
                    data_threshold = np.percentile(flat_data, 90)
                    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

                    # Plot for predicted data
                    contourf0 = axs[0].contourf(X, Y, data, levels=50, cmap='viridis')  # Use 50 levels for detailed gradients

                    # Draw the threshold contour if it exists in the data range
                    if np.min(data) <= data_threshold <= np.max(data):
                        contour0 = axs[0].contour(X, Y, data, levels=[data_threshold], colors='red')
                        # Add the threshold value as a label to the contour line
                        axs[0].clabel(contour0, inline=True, fontsize=8)

                    axs[0].set_title(f'Contour Plot of Predicted {variable} at last time step for Sample {sample+1}')
                    axs[0].set_xlabel('x')
                    axs[0].set_ylabel('y')

                    # Plot for true data
                    contourf1 = axs[1].contourf(X, Y, true_data, levels=50, cmap='viridis')  # Use 50 levels for detailed gradients

                    # Draw the threshold contour if it exists in the data range
                    if np.min(true_data) <= data_threshold <= np.max(true_data):
                        contour1 = axs[1].contour(X, Y, true_data, levels=[data_threshold], colors='red')
                        # Add the threshold value as a label to the contour line
                        axs[1].clabel(contour1, inline=True, fontsize=8)

                    axs[1].set_title(f'Contour Plot of True {variable} at last time step for Sample {sample+1}')
                    axs[1].set_xlabel('x')
                    axs[1].set_ylabel('y')

                    plt.tight_layout()
                    plt.colorbar(contourf1)  # Use the colorbar from the true data plot
                    plt.savefig(os.path.join(image_folder, f"contour_{sample+1}.png"))
                    plt.show()
                    plt.close()

                    try: 
                        fig, ax = plt.subplots(figsize=(9, 7))

                        max_val_pos_data = np.unravel_index(data.argmax(), data.shape)
                        max_val_pos_true_data = np.unravel_index(true_data.argmax(), true_data.shape)


                        # Draw the background data (true data)
                        contourf = ax.imshow(true_data, cmap='viridis', alpha=0.5, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')

                        # Draw the contour lines for the predicted and true data
                        contour_pred = ax.contour(X, Y, data, levels=[data_threshold], colors='red')
                        contour_true = ax.contour(X, Y, true_data, levels=[data_threshold], colors='blue')

                        # Create a legend manually with proxy artists
                        import matplotlib.lines as mlines
                        # Draw the contour lines for the predicted and true data
                        contour_pred = ax.contour(X, Y, data, levels=[data_threshold], colors='red')
                        contour_true = ax.contour(X, Y, true_data, levels=[data_threshold], colors='blue')

                        # Create a legend manually with patches
                    
                        red_patch = mpatches.Patch(color='red', label='Predicted')
                        blue_patch = mpatches.Patch(color='blue', label='True')
                        ax.legend(handles=[red_patch, blue_patch])


                        # Compute the distances and display them as annotations
                        dist_data = np.min(np.sqrt((max_val_pos_data[0]-contour_pred.allsegs[0][0][:,1])**2 + (max_val_pos_data[1]-contour_pred.allsegs[0][0][:,0])**2))
                        dist_true_data = np.min(np.sqrt((max_val_pos_true_data[0]-contour_true.allsegs[0][0][:,1])**2 + (max_val_pos_true_data[1]-contour_true.allsegs[0][0][:,0])**2))
                        ax.annotate(f'Min Distance (Pred): {dist_data:.2f}', (0.05, 0.95), xycoords='axes fraction', backgroundcolor='white')
                        ax.annotate(f'Min Distance (True): {dist_true_data:.2f}', (0.05, 0.85), xycoords='axes fraction', backgroundcolor='white')
                        predicted_distances.append(dist_data)
                        true_distances.append(dist_true_data)

                        # Set the title
                        ax.set_title('Overlay of True and Predicted Contours')

                        plt.tight_layout()
                        plt.colorbar(contourf, ax=ax)  # Use the colorbar from the true data plot
                        plt.savefig(os.path.join(image_folder, f"contour_overlay_{sample+1}.png"))
                        plt.show()

                    except Exception as e:
                        print(f"An error occurred: {e}")


            
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
plot_histogram_distances =True
# if plot_histogram_distances:
#     for batch_idx, (x, y) in enumerate(test_loader): 

#         #check if batch_idx is in samples to plot
#         x = x.to(device)
#         true_y = y.to(device)
#         x = input_normalizer.encode(x)
#         out = model(x)
#         out = output_normalizer.decode(out)
#         num_samples = x.size(0)          
#         for index in range(BATCH_SIZE):
#             sample = batch_idx * BATCH_SIZE + index
#             test_y = true_y[index,...].detach().cpu()
#             predicted_y = out[index,...].detach().cpu()
#             data = predicted_y[-1, :, :, 0].numpy()
#             true_data = test_y[-1, :, :, 0].numpy()
            
#             X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
#             flat_data  = data.flatten()
#             data_threshold = np.percentile(flat_data, 90)    


#             max_val_pos_data = np.unravel_index(data.argmax(), data.shape)
#             max_val_pos_true_data = np.unravel_index(true_data.argmax(), true_data.shape)
#             # Draw the contour lines for the predicted and true data
#             contour_pred = ax.contour(X, Y, data, levels=[data_threshold], colors='red')
#             contour_true = ax.contour(X, Y, true_data, levels=[data_threshold], colors='blue')
#             dist_data = np.min(np.sqrt((max_val_pos_data[0]-contour_pred.allsegs[0][0][:,1])**2 + (max_val_pos_data[1]-contour_pred.allsegs[0][0][:,0])**2))
#             dist_true_data = np.min(np.sqrt((max_val_pos_true_data[0]-contour_true.allsegs[0][0][:,1])**2 + (max_val_pos_true_data[1]-contour_true.allsegs[0][0][:,0])**2))
            
#             predicted_distances.append(dist_data)
#             true_distances.append(dist_true_data)
# #%%
# # Create a histogram of predicted distances
# plt.hist(predicted_distances, bins=20, alpha=0.5, label='Predicted Distances')

# # Create a histogram of true distances
# plt.hist(true_distances, bins=20, alpha=0.5, label='True Distances')

# # Set the labels and title of the plot
# plt.xlabel('Max Distance')
# plt.ylabel('Frequency')
# plt.title('Histogram of Predicted and True Max Distances')

# # Add a legend
# plt.legend()

# # Show the plot
# plt.show()
# %%
