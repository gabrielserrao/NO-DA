#%%
"""
Adapted from Zongyi Li TODO: include referene in the README
This file is the Fourier Neural Operator for 3D problem takes the 2D spatial + 1D temporal equation directly as a 3D problem
"""

import torch.nn.functional as F
from utilities import *
from timeit import default_timer
import matplotlib.pyplot as plt
import numpy as np
import resource




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

################################################################
# configs-1
################################################################
#create a variable called resolution 
resolution = 32
folder = "/scratch/smrserraoseabr/Projects/FluvialCO2/results" + str(resolution) + "/"
input_vars = ['Por', 'Perm', 'gas_rate'] # Porosity, Permeability, ,  Well 'gas_rate', Pressure + x, y, time encodings 
output_vars = ['Pressure'] 




num_files= 100
traintest_split = 0.8

batch_size = 16

ntrain = num_files*traintest_split
ntest = num_files - ntrain

learning_rate = 0.001
epochs = 80


iterations = epochs*(ntrain//batch_size)
modes = 12
width = 128 

# Prepare the path
path = 'ns_fourier_3d_N{}_ep{}_m{}_w{}'.format(ntrain, epochs, modes, width)

# Include in the path the input and output variables
path += '_INPUT_' + '_'.join(input_vars) + '_OUTPUT_' + '_'.join(output_vars)

# Create paths for log, model, and images
path_log = os.path.join('runs', path, 'log')
# Modify here: 'model.pt' will be the filename, not a subdirectory
path_model = os.path.join('runs', path, 'model.pt') 
path_image = os.path.join('runs', path, 'images')

# Create directories
os.makedirs(path_log, exist_ok=True)
os.makedirs(os.path.dirname(path_model), exist_ok=True)  # Get the directory of the path_model
os.makedirs(path_image, exist_ok=True)

# Create paths for train error and test error files
path_train_err = os.path.join(path_log, 'train.txt')
path_test_err = os.path.join(path_log, 'test.txt')



S = 32
#T_in = 61
T = 61

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#device = 'cpu'
print('Using ' + device + ' for training')

################################################################
# load data
################################################################

################################################################
# load data
################################################################
runtime = np.zeros(2, )
t1 = default_timer()

# Create instance of ReadXarrayDatasetNorm class for training data
dataset = ReadXarrayDataset(folder=folder, input_vars=input_vars, output_vars=output_vars, num_files = num_files)

train_size = int(traintest_split * len(dataset))
test_size = len(dataset) - train_size

train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
test_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
t2 = default_timer()

# We no longer have the entire dataset loaded into memory. The normalization is handled by the Dataset class.
# We also can't directly print the shape of the data because it's not all loaded into memory.



# Normalize input_data and output_data
# Create normalizers for training data
train_input_normalizer = PointGaussianNormalizer(train_loader, is_label=False)
train_output_normalizer = PointGaussianNormalizer(train_loader, is_label=True)

# Create normalizers for test data
test_input_normalizer = PointGaussianNormalizer(test_loader, is_label=False)
test_output_normalizer = PointGaussianNormalizer(test_loader, is_label=True)

train_input_normalizer = train_input_normalizer.cuda(device)
train_output_normalizer = train_output_normalizer.cuda(device)

test_input_normalizer = test_input_normalizer.cuda(device)
test_output_normalizer = test_output_normalizer.cuda(device)

print(f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
print('preprocessing finished, time used:', t2-t1)



#%%
################################################################
# training and evaluation
################################################################
model = FNO3d(modes, modes, modes, width).to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    print(f'epoch {ep} of {epochs}')
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        
        x = x.to(device)
        y = y.to(device)
        
        x = train_input_normalizer.encode(x)
        y = train_output_normalizer.encode(y) 

        optimizer.zero_grad()
        out = model(x) #.view(batch_size, S, S, T)

        mse = F.mse_loss(out, y, reduction='mean')
        # mse.backward()

        y = train_output_normalizer.decode(y)
        out = train_output_normalizer.decode(out)
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()

        optimizer.step()
        scheduler.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    model.eval()
    test_l2 = 0.0
    test_mse= 0.0
    with torch.no_grad():
        for index, (x, y) in enumerate(test_loader):
            
            x = x.to(device)
            y = y.to(device)

            x = test_input_normalizer.encode(x)
            y = test_output_normalizer.encode(y)
            

            out = model(x) #.view(batch_size, S, S, T)
            mse = F.mse_loss(out, y, reduction='mean')

            y = test_output_normalizer.decode(y)
            out = test_output_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()            
            test_mse += mse.item()

            if index == 0:
                test_y_shape = (1, 61, resolution, resolution, 1)
                predicted_y_shape = (1, 61, resolution, resolution, 1)
                test_y = y[0].detach().view(test_y_shape).cpu().numpy()
                predicted_y = out[0].detach().view(predicted_y_shape).cpu().numpy()
                fig, ax = plt.subplots(nrows=1, ncols=3)
                ax[0].imshow(test_y[0, -1, :, :, 0].T)
                ax[1].imshow(predicted_y[0, -1, :, :, 0].T)
                ax[2].imshow((test_y[0, 0, :, :, 0]-predicted_y[0, 0, :, :, 0]).T)
                plt.savefig(path_image + '_ep' + str(ep) + '.png')
                plt.close()
                #detached_out = out.detach().cpu().numpy()
       
#which is the correct command to submit sbatch?
#sbatch -p gpu --gres=gpu:1 --mem=10000 --time=0-00:30:00 --wrap="python fouier_3d_GPU_enabled.py"
    

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_mse/= len(test_loader)
    test_l2 /= ntest

    t2 = default_timer()
    #print mse, l2 for train and test data for each epoch
    print(f'ep {ep}: t={t2-t1:.3f}, train_mse={train_mse:.3e}, train_l2={train_l2:.3e}, test_l2={test_l2:.3e}, test_mse={test_mse:.3e}')
    #save train and test mse, l2 for each epoch
    with open(path_train_err, 'a') as f:
        f.write(f'epoch {ep}: t={t2-t1:.3f}, train_mse={train_mse:.3e}, train_l2={train_l2:.3e}\n')
    with open(path_test_err, 'a') as f:
        f.write(f'epoch {ep}: t={t2-t1:.3f}, test_mse={test_mse:.3e}, test_l2={test_l2:.3e}\n') 

    #for each 10 epochs, save the model
    if ep % 10 == 0:
        torch.save(model, path_model)
    
torch.save(model, path_model)
#%%
#pred = torch.zeros(test_u.shape)
# index = 0
# #test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
# with torch.no_grad():
#     for x, y in test_loader:
#         test_l2 = 0
#         x.to(device) 
#         y.to(device) 

#         out = model(x)
#         out = test_output_normalizer.decode(out)
#         y = test_output_normalizer.decode(y)
#         #pred[index] = out

#         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#         print(index, test_l2)
#         index = index + 1
#         #plot 

#scipy.io.savemat(os.path.join('runs', path, 'log', path+'.mat'), mdict={'pred': pred.cpu().numpy()})


###

