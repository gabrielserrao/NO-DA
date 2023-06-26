# %%
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

# %%


# DATASET
data_folder = "/scratch/smrserraoseabr/Projects/FluvialCO2/results32/"
num_files = 1000
traintest_split = 0.8
num_samples = 10  # Number of samples to see results
#CASE CONFIGURATION
input_vars = ['Por', 'Perm', 'gas_rate'] # Porosity, Permeability, ,  Well 'gas_rate', Pressure + x, y, time encodings 
variable = 'Pressure'
output_vars = [variable]
case_name = 'VINI_ns_fourier_3d_N800.0_ep500_m12_w128_b61_padding4' + '_' + variable
#DEVICE SETTINGS
device = 'cpu'
#OUTPUT CONFIGURATION
plot_model_eval =True
plot_comparison = True
plot_lines = True
plot_gifs =True


###############################################



path_runs = os.path.join('runs', case_name)
path_model = os.path.join(path_runs, 'model', f'{case_name}_model.pt')
path_normalizer = os.path.join(path_runs, 'model')
image_folder = os.path.join(path_runs, 'images')
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
      


# %%
#LOAD NORMALIZATION PARAMETERS
dataset = ReadXarray(folder=data_folder, input_vars=input_vars, output_vars=output_vars, num_files = num_files, traintest_split = traintest_split)

# Get input and output data tensors
train_a = dataset.train_data_input
train_u = dataset.train_data_output

test_a = dataset.test_data_input
test_u = dataset.test_data_output

print(f'Input train data shape: {train_a.shape}')
print(f'Output train data shape: {train_u.shape}')

print(f'Input test data shape: {test_a.shape}')
print(f'Output test data shape: {test_u.shape}')


# Move data tensors to GPU if available
train_a = train_a.to(device)
train_u = train_u.to(device)

test_a = test_a.to(device)
test_u = test_u.to(device)

# Normalize input_data and output_data
#check if normalizer exists and load it, otherwise raise a warning and create a new one
if os.path.exists(os.path.join((path_normalizer), f'{case_name}_a_normalizer_mean.pt')):
    a_normalizer_mean = torch.load(os.path.join((path_normalizer), f'{case_name}_a_normalizer_mean.pt'))
    a_normalizer_std = torch.load(os.path.join((path_normalizer), f'{case_name}_a_normalizer_std.pt'))
    y_normalizer_mean = torch.load(os.path.join((path_normalizer), f'{case_name}_y_normalizer_mean.pt'))
    y_normalizer_std = torch.load(os.path.join((path_normalizer), f'{case_name}_y_normalizer_std.pt'))
    print('Normalizer loaded')
    a_normalizer = UnitGaussianNormalizer(train_a, mean=a_normalizer_mean, std=a_normalizer_std)
    y_normalizer = UnitGaussianNormalizer(train_u, mean=y_normalizer_mean, std=y_normalizer_std)


else:
    print('Normalizer not found, creating a new one and saving it')
    a_normalizer = UnitGaussianNormalizer(train_a)
    y_normalizer = UnitGaussianNormalizer(train_u)
    torch.save(a_normalizer.mean, os.path.join((path_normalizer), f'{case_name}_a_normalizer_mean.pt'))
    torch.save(a_normalizer.std, os.path.join((path_normalizer), f'{case_name}_a_normalizer_std.pt'))

    torch.save(y_normalizer.mean, os.path.join((path_normalizer), f'{case_name}_y_normalizer_mean.pt'))
    torch.save(y_normalizer.std, os.path.join((path_normalizer), f'{case_name}_y_normalizer_std.pt'))


train_a= a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

train_u = y_normalizer.encode(train_u)
test_u = y_normalizer.encode(test_u)

#DEFINE TRUE DATA - RESULTS OF DARTS SIMULATION
true = y_normalizer.decode(test_u)


# %%
#Load model and print summary
model = torch.load(path_model)
model.eval(); 
print(model)
#extract main parameters of model and print them like padding, number of layers, etc
print('Model loaded')
#print number of parameters of model
print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')


# %%
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

    path_normalizer
    
   # Extract data
    epoch_train, mse_train, l2_train = extract_data(os.path.join(path_normalizer, 'train.txt'))

    epoch_test, mse_test, l2_test =  extract_data(os.path.join(path_normalizer, 'test.txt'))

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
    plt.savefig(os.path.join(path_runs, 'model', f'{case_name}_model_eval_L2.png'))

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
    plt.savefig(os.path.join(path_runs, 'model', f'{case_name}_model_eval_MSE.png'))




#save figure on model folder
   
# Show plot
plt.show()

# %%
#predict number of samples
test_a = test_a[:num_samples]
pred = model(test_a)
pred_un = y_normalizer.decode(pred)

# %%
if plot_comparison:
    for index in range(num_samples):
        test_y = true[index,...].detach().numpy()
        predicted_y = pred_un[index,...].detach().numpy()

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
        norm = mpl.colors.Normalize(vmin=colorbar_vmin, vmax=colorbar_vmax)

        img1 = axes[0].imshow(test_y[-1, :, :, 0], cmap='jet', norm=norm)
        img2 = axes[1].imshow(predicted_y[-1, :, :, 0], cmap='jet', norm=norm)
        img3 = axes[2].imshow(np.abs(test_y[ -1, :, :, 0] - predicted_y[ -1, :, :, 0]), cmap='jet')

        for img, ax in zip([img1, img2], axes[:2]):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(img, cax=cax, orientation='vertical')

        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img3, cax=cax, orientation='vertical')

        axes[0].set_title(f'Test - Sample {index+1}')
        axes[1].set_title(f'Predicted  - Sample {index+1}')
        axes[2].set_title(f'Absolute Error - Sample {index+1}')

        for ax in axes:
            ax.axis('off')

        fig.suptitle(f'Comparison of Test and Predicted {variable} values for 3D Fourier Neural Operator')
        plt.tight_layout()
        plt.savefig(os.path.join(image_folder, f"comparison_{index+1}.png"))
        plt.show()
        

# %%
# Given x and y arrays

if plot_lines:
    resolution  = test_y.shape[1]
    x = np.arange(0, resolution, 4)
    y = np.arange(0, resolution, 4)

    time = test_y[:,0,0,0]

    # Initialize empty lists for each line
    diagonal = []
    x_line = []
    y_line = []

    # Iterating over all elements
    for i in range(len(x)):
        for j in range(len(y)):
            # For the diagonal, x and y are the same
            if i == j:
                diagonal.append((x[i], y[j]))
            # For the x line, x equals 32 / 2
            if x[i] == 32 / 2:
                x_line.append((x[i], y[j]))
            # For the y line, y equals 32 / 2
            if y[j] == 32 / 2:
                y_line.append((x[i], y[j]))

    # Now you have your three lists:
    print("Diagonal: ", diagonal)
    print("X Line: ", x_line)
    print("Y Line: ", y_line)
    for index in range(num_samples):
        test_y = true[index,...].detach().numpy()
        predicted_y = pred_un[index,...].detach().numpy()

        for line in [diagonal, x_line, y_line]:
            for i, (x, y) in enumerate(line):

                # Create main figure and axis
                fig, main_ax = plt.subplots()

                # Plot your data on the main axes
                main_ax.plot(time, test_y[:, x, y, 0], label='True', linestyle='solid', color = 'blue')
                main_ax.plot(time, predicted_y[:, x, y, 0], label='Predicted', linestyle='none', marker='o', color = 'red')
                main_ax.legend()

                # Set labels and title for the main figure
                main_ax.set_xlabel('Time')
                main_ax.set_ylabel(variable)
                main_ax.set_title(f'Sample {index} {variable} at x= {x} and y = {y} ')

                # Create inset of width 30% and height 30% at the upper right corner of the main plot
                left, bottom, width, height = [0.15, 0.45, 0.3, 0.3] # adjust as needed
                inset_ax = fig.add_axes([left, bottom, width, height])
                im = inset_ax.imshow(test_a[0,-1, :, :, 1], cmap='jet')

                inset_ax.scatter(x, y, s=20, edgecolor='black', facecolor='none', linewidth=2) # Assuming the step size is 4

                #fix scale of main plot between 0 and 1 
                main_ax.set_ylim([colorbar_vmin, colorbar_vmax])

                inset_ax.axis('off')
                #save the figure with a name that includes the x and y coordinates and the variable name on the images folder

                fig.savefig(os.path.join(image_folder, f'Sample_{index+1}_comparison_point_x{x}_y{y}.png'), dpi=300)

                # Show the plot
                plt.show()

                # Closing the figure to prevent from running out of memory
                plt.close(fig)



# %%
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

if plot_gifs == True:

    gif_paths = []

    for sample in range(num_samples):
        # Rest of the code to generate the GIF for each sample
        image_buffers = []
        test_y = true[sample,...].detach().numpy()
        predicted_y = pred_un[sample,...].detach().numpy()

        for t in range(61):
            buf = plot_to_memory_image(test_y[t, :, :, 0], predicted_y[t, :, :, 0], t, variable = variable)
            image_buffers.append(buf)

        images = [imageio.imread(buf.getvalue()) for buf in image_buffers]
        buf_result = BytesIO()
        imageio.mimsave(buf_result, images, format='GIF', duration=0.5)

        # Save the GIF
        gif_save_path = os.path.join(image_folder, f'{case_name}_{sample}.gif')
        with open(gif_save_path, 'wb') as f:
            f.write(buf_result.getvalue())
            buf_result.seek(0)

        for buf in image_buffers:
            buf.close()

        # Store the GIF path in the list
        gif_paths.append(gif_save_path)

    image_buffers = []
    test_y = true[sample,...].detach().numpy()
    predicted_y = pred_un[sample,...].detach().numpy()

    for t in range(61):
        buf = plot_to_memory_image(test_y[t, :, :, 0], predicted_y[t, :, :, 0], t, variable = variable)
        image_buffers.append(buf)

    images = [imageio.imread(buf.getvalue()) for buf in image_buffers]
    buf_result = BytesIO()
    imageio.mimsave(buf_result, images, format='GIF', duration=0.5)

    gif_save_path = os.path.join(image_folder, f'{case_name}_{sample+1}.gif')
    with open(gif_save_path, 'wb') as f:
        f.write(buf_result.getvalue())

    buf_result.seek(0)

    for buf in image_buffers:
        buf.close()

    DisplayImage(buf_result.getvalue(), format='png')

# %%



