#%%
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

import torch.nn.functional as F
from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)
#%%
"""
Adapted from Zongyi Li TODO: include referene in the README
This file is the Fourier Neural Operator for 3D problem takes the 2D spatial + 1D temporal equation directly as a 3D problem
"""

import torch.nn.functional as F
from utilities import *
from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)
#%%
################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: all parameters  + encoded spatial-temporal locations (x, y, t)
        input shape: (batchsize, x=32, y=32, t=61, c=6)
        output: the solution of the 61 timesteps
        output shape: (batchsize, x=32, y=32, t=61, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 4 # pad the domain if input is non-periodic -. defautl 4 
        #TODO: padding = 4 

        self.p = nn.Linear(6, self.width)# input channel is 7: Por, Perm, gas_rate, Pressure + x, y, time encodings
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x): # (batchsize, x=32, y=32, t=61, c=6)
        grid = self.get_grid(x.shape, x.device)
        #print(f'grid shape: {grid.shape}')
        #print(f'x shape: {x.shape}')
        #x = torch.cat((x, grid), dim=-1)
        #print(f'x shape after cat: {x.shape}')
        x = self.p(x) # output size: batchsize, channel , width
        x = x.permute(0, 4, 1, 2, 3)

        # TODO: modificar o padding para que ajustar x, y e t  
        #p3d -> x, y, t
        p3d = (self.padding, self.padding, self.padding, self.padding, self.padding, self.padding)
        x = F.pad(x, p3d) # pad the domain if input is non-periodic
        #x = F.pad(x, [0,self.padding]) # ORIGINAL
        x1 = self.conv0(x) #Fourier layer
        x1 = self.mlp0(x1) #Conv layer (input fourier layer output)
        x2 = self.w0(x) #Conv layer (input x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        #x = x[..., :-self.padding] ORIGINAL
        #retirar o p3d referente aos ultimos 3 indices
        x = x[..., self.padding:-self.padding, self.padding:-self.padding, self.padding:-self.padding] 
        
        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        return x


    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
        
      

# define all hardcoded variables and paths here
folder = "/scratch/smrserraoseabr/Projects/FluvialCO2/results32/"
input_vars = ['Por', 'Perm', 'gas_rate'] # Porosity, Permeability, ,  Well 'gas_rate', Pressure + x, y, time encodings 
variable = 'CO_2'
output_vars = [variable] 
device = 'cpu'
num_files = 1000
traintest_split = 0.8
path_runs = f'runs/OK_ns_fourier_3d_N800.0_ep100_m12_w128_b61_padding4_{variable}'
model_name = f'ns_fourier_3d_N800.0_ep100_m12_w128_b61_padding4_{variable}'
path_model = os.path.join(path_runs, 'model', f'{model_name}.pt')
#path_model = '/scratch/smrserraoseabr/Projects/NO-DA/runs/OK_ns_fourier_3d_N800.0_ep100_m12_w128_b61_padding4_CO2/model/ns_fourier_3d_N800.0_ep100_m12_w128_b61_padding4_CO2.pt'
num_samples = 200  # Number of samples to iterate over
sample = 5
if variable == 'CO_2':
    colorbar_vmax, colorbar_vmin = 0.0, 1.0 # Define your min and max here
elif variable == 'Pressure':
    colorbar_vmin, colorbar_vmax = 200.0, 600.0 # Define your min and max here
  # Change this to the index you want
image_folder = os.path.join(path_runs, 'images')
#image_folder = '/scratch/smrserraoseabr/Projects/NO-DA/runs/OK_ns_fourier_3d_N800.0_ep100_m12_w128_b61_padding4_CO2/images'

# Create image_folder if it doesn't exist
os.makedirs(image_folder, exist_ok=True)

# Now, use the variables and paths in the rest of your code
dataset = ReadXarray(folder=folder, input_vars=input_vars, output_vars=output_vars, num_files = num_files, traintest_split = traintest_split)

model = torch.load(path_model)
model.eval()

train_a = dataset.train_data_input
train_u = dataset.train_data_output
test_a = dataset.test_data_input
test_u = dataset.test_data_output

train_a = train_a.to(device)
train_u = train_u.to(device)
test_a = test_a.to(device)
test_u = test_u.to(device)

a_normalizer = UnitGaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

y_normalizer = UnitGaussianNormalizer(train_u)
train_u = y_normalizer.encode(train_u)
test_u = y_normalizer.encode(test_u)

true = y_normalizer.decode(test_u)

pred = model(test_a)
pred_un = y_normalizer.decode(pred)

for index in range(num_samples):
    test_y = true[index,...].detach().numpy()
    predicted_y = pred_un[index,...].detach().numpy()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))
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

    axes[0].set_title(f'Test y - Sample {index+1}')
    axes[1].set_title(f'Predicted y - Sample {index+1}')
    axes[2].set_title(f'Absolute Error - Sample {index+1}')

    for ax in axes:
        ax.axis('off')

    fig.suptitle(f'Comparison of Test and Predicted {variable} % values for 3D Fourier Neural Operator')
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, f"comparison_{index+1}.png"))
    plt.close()

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
    gif_save_path = os.path.join(image_folder, f'{model_name}_{sample}.gif')
    with open(gif_save_path, 'wb') as f:
        f.write(buf_result.getvalue())
        buf_result.seek(0)

    for buf in image_buffers:
        buf.close()
    
    # Store the GIF path in the list
    gif_paths.append(gif_save_path)

# image_buffers = []
# test_y = true[sample,...].detach().numpy()
# predicted_y = pred_un[sample,...].detach().numpy()

# for t in range(61):
#     buf = plot_to_memory_image(test_y[t, :, :, 0], predicted_y[t, :, :, 0], t, variable = variable)
#     image_buffers.append(buf)

# images = [imageio.imread(buf.getvalue()) for buf in image_buffers]
# buf_result = BytesIO()
# imageio.mimsave(buf_result, images, format='GIF', duration=0.5)

# gif_save_path = os.path.join(image_folder, f'{model_name}_{sample}.gif')
# with open(gif_save_path, 'wb') as f:
#     f.write(buf_result.getvalue())

# buf_result.seek(0)

# for buf in image_buffers:
#     buf.close()




#DisplayImage(buf_result.getvalue(), format='png')
# %%
