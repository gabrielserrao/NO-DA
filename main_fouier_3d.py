#%%
"""
Adapted from Zongyi Li TODO: include referene in the README
This file is the Fourier Neural Operator for 3D problem takes the 2D spatial + 1D temporal equation directly as a 3D problem
"""
import sys
import os
import torch.nn.functional as F
from utilities import *
from timeit import default_timer
import matplotlib.pyplot as plt
import numpy as np
import resource
from model_fourier_3d import *
import torch
print(torch.__version__)
print(f"GPUs:{torch.cuda.device_count()}")
import os
print(os.getcwd())
#%%
################################################################
# configs-1
################################################################
torch.manual_seed(1)
np.random.seed(1)

tag = 'MonthQgWellCenter' 
folder = '/samoa/data/smrserraoseabr/NO-DA/dataset/DARTS/runnedmodels_wells/filtered' # '/samoa/data/smrserraoseabr/NO-DA/dataset/mixedcontext32x32' #"/samoa/data/smrserraoseabr/NO-DA/dataset/DARTS/runnedmodels_wells/filtered"  #  "/nethome/atena_projetos/bgy3/NO-DA/datasets/results" + str(resolution) + "/"
input_vars = ['Por', 'Perm', 'gas_rate'] # Porosity, Permeability, ,  Well 'gas_rate', Pressure + x, y, time encodings 
output_vars = ['CO_2'] 
num_files= 1000
traintest_split = 0.8
batch_size = 10
normalizer = 'PointGaussianNormalizerNoNaN'
WELLS_POSITIONS = True
learning_rate = 0.001
epochs = 300
modes = 18
width = 128

ntrain = num_files*traintest_split
ntest = num_files - ntrain
iterations = epochs*(ntrain//batch_size)


# Prepare the path
path = 'FNO_3d_{}_N{}_ep{}_m{}_w{}_b{}_norm{}'.format(tag,ntrain, epochs, modes, width, batch_size, normalizer)


# Include in the path the input and output variables
path += '_INPUT_' + '_'.join(input_vars) + '_OUTPUT_' + '_'.join(output_vars)

# Create paths for log, model, and images
path_log = os.path.join('runs', path, 'log')
# Modify here: 'model.pt' will be the filename, not a subdirectory
path_model = os.path.join('runs', path, f'{path}_model.pt') 
path_image = os.path.join('runs', path, 'images')

# Create directories
os.makedirs(path_log, exist_ok=True)
os.makedirs(os.path.dirname(path_model), exist_ok=True)  # Get the directory of the path_model
os.makedirs(path_image, exist_ok=True)

# Create paths for train error and test error files
path_train_err = os.path.join(path_log, 'train.txt')
path_test_err = os.path.join(path_log, 'test.txt')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print('Using ' + device + ' for training')

#%%
################################################################
# load data
################################################################
runtime = np.zeros(2, )
t1 = default_timer()

# Create instance of ReadXarrayDatasetNorm class for training data
dataset = ReadXarrayDataset(folder=folder, input_vars=input_vars, output_vars=output_vars, num_files = num_files, wells_positions=WELLS_POSITIONS)

train_size = int(traintest_split * len(dataset))
test_size = len(dataset) - train_size


train_loader = DataLoader(torch.utils.data.Subset(dataset, range(0, train_size)),
                           batch_size=batch_size,
                             shuffle=False)
test_loader = DataLoader(torch.utils.data.Subset(dataset, range(train_size, train_size + test_size)), 
                         batch_size=batch_size, 
                         shuffle=False)

t2 = default_timer()
#%%
# We no longer have the entire dataset loaded into memory. The normalization is handled by the Dataset class.

if normalizer == 'PointGaussian':
    input_normalizer = PointGaussianNormalizer(train_loader, is_label=False)
    output_normalizer = PointGaussianNormalizer(train_loader, is_label=True)

    input_normalizer = input_normalizer.cuda(device)
    output_normalizer = output_normalizer.cuda(device)


    #save the normalizers mean and std on pytorch files
    torch.save(input_normalizer.mean,os.path.join('runs', path, 'normalizer_mean_input.pt'))
    torch.save(input_normalizer.std, os.path.join('runs', path, 'normalizer_std_input.pt'))
    torch.save(output_normalizer.mean, os.path.join('runs', path, 'normalizer_mean_output.pt'))
    torch.save(output_normalizer.std, os.path.join('runs', path, 'normalizer_std_output.pt'))

elif normalizer == 'Gaussian':  
    input_normalizer = GaussianNormalizer(train_loader, is_label=False)
    output_normalizer = GaussianNormalizer(train_loader, is_label=True)

    input_normalizer = input_normalizer.cuda(device)
    output_normalizer = output_normalizer.cuda(device)

    #save the normalizers mean and std on pytorch files
    torch.save(input_normalizer.mean,os.path.join('runs', path, 'normalizer_mean_input.pt'))
    torch.save(input_normalizer.std, os.path.join('runs', path, 'normalizer_std_input.pt'))
    torch.save(output_normalizer.mean, os.path.join('runs', path, 'normalizer_mean_output.pt'))
    torch.save(output_normalizer.std, os.path.join('runs', path, 'normalizer_std_output.pt'))
    
elif normalizer == 'PointGaussianNormalizerNoNaN':
    input_normalizer = PointGaussianNormalizerNoNaN(train_loader, is_label=False)
    output_normalizer = PointGaussianNormalizerNoNaN(train_loader, is_label=True)

    input_normalizer = input_normalizer.cuda(device)
    output_normalizer = output_normalizer.cuda(device)


    #save the normalizers mean and std on pytorch files
    torch.save(input_normalizer.mean,os.path.join('runs', path, 'normalizer_mean_input.pt'))
    torch.save(input_normalizer.std, os.path.join('runs', path, 'normalizer_std_input.pt'))
    torch.save(output_normalizer.mean, os.path.join('runs', path, 'normalizer_mean_output.pt'))
    torch.save(output_normalizer.std, os.path.join('runs', path, 'normalizer_std_output.pt'))


print_memory_usage()
print('preprocessing finished, time used:', t2-t1)



#%%
################################################################
# training and evaluation
################################################################
model = FNO3d(modes, modes, modes, width)
print_memory_usage()
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
        
        x = x.to(device) #(batch, 61, 32, 32, 6)
        y = y.to(device) #(batch, 61, 32, 32, 1)
        
        x = input_normalizer.encode(x)
        y = output_normalizer.encode(y) 

        optimizer.zero_grad()
        out = model(x)

        mse = F.mse_loss(out, y, reduction='mean')
        # mse.backward()

        y = output_normalizer.decode(y)
        out = output_normalizer.decode(out)
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()

        optimizer.step()
        scheduler.step()
        train_mse += mse.item()
        train_l2 += l2.item() 
           
        #print_memory_usage()

    model.eval()
    test_l2 = 0.0
    test_mse= 0.0
    with torch.no_grad():
        for index, (x, y) in enumerate(test_loader):
            
            x = x.to(device)
            y = y.to(device)

            x = input_normalizer.encode(x)
            y = output_normalizer.encode(y)
            

            out = model(x) #.view(batch_size, S, S, T)
            mse = F.mse_loss(out, y, reduction='mean')

            y = output_normalizer.decode(y)
            out = output_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()            
            test_mse += mse.item()

            #save figure every 10 epochs
            if ep % 10 == 0 and index == 0:
                test_y_shape = (1, 61, 32, 32, 1)
                predicted_y_shape = (1, 61, 32, 32, 1)
                test_y = y[0].detach().view(test_y_shape).cpu().numpy()
                predicted_y = out[0].detach().view(predicted_y_shape).cpu().numpy()
                fig, ax = plt.subplots(nrows=1, ncols=3)
                ax[0].imshow(test_y[0, -1, :, :, 0].T)
                ax[1].imshow(predicted_y[0, -1, :, :, 0].T)
                ax[2].imshow((test_y[0, 0, :, :, 0]-predicted_y[0, 0, :, :, 0]).T)
                plt.savefig(os.path.join(path_image, 'test_ep' + str(ep) + '.png'))
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

# %%
