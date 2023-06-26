#%%
"""
Adapted from Zongyi Li TODO: include referene in the README
This file is the Fourier Neural Operator for 3D problem takes the 2D spatial + 1D temporal equation directly as a 3D problem
"""
#print version of python and available packages
import sys
import os
import torch.nn.functional as F
from utilities import *
from timeit import default_timer
import matplotlib.pyplot as plt
import numpy as np
import resource
from model_fourier_3d import *

#print torch version
import torch
print(torch.__version__)
print(f"GPUs:{torch.cuda.device_count()}")

torch.manual_seed(0)
np.random.seed(0)


################################################################
# configs-1
################################################################
#create a variable called resolution 
resolution = 32

folder = "/nethome/atena_projetos/bgy3/NO-DA/datasets/results" + str(resolution) + "/"
input_vars = ['Por', 'Perm', 'gas_rate'] # Porosity, Permeability, ,  Well 'gas_rate', Pressure + x, y, time encodings 
output_vars = ['Pressure'] 



num_files= 1000
traintest_split = 0.8

batch_size = 1

ntrain = num_files*traintest_split
ntest = num_files - ntrain

learning_rate = 0.001
epochs = 100


iterations = epochs*(ntrain//batch_size)
modes = 12
width = 64 

# Prepare the path
path = 'FNO_3d_N{}_ep{}_m{}_w{}_b{}'.format(ntrain, epochs, modes, width, batch_size)

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
input_normalizer = PointGaussianNormalizer(train_loader, is_label=False)
output_normalizer = PointGaussianNormalizer(train_loader, is_label=True)

# Create normalizers for test data

train_input_normalizer = input_normalizer.cuda(device)
train_output_normalizer = output_normalizer.cuda(device)

test_input_normalizer = input_normalizer.cuda(device)
test_output_normalizer = output_normalizer.cuda(device)

#save the normalizers mean and std on pytorch files
torch.save(train_input_normalizer.mean, os.path.join(path_model, 'mean_input.pt'))
torch.save(train_input_normalizer.std, os.path.join(path_model, 'std_input.pt'))
torch.save(train_output_normalizer.mean, os.path.join(path_model, 'mean_output.pt'))
torch.save(train_output_normalizer.std, os.path.join(path_model, 'std_output.pt'))


print(f"Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
print('preprocessing finished, time used:', t2-t1)



#%%
################################################################
# training and evaluation
################################################################
model = FNO3d(modes, modes, modes, width)


model.to(device)
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
