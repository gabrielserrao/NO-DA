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
from model_fourier_3d import *
################################################################
# configs-1
################################################################
folder = "/scratch/smrserraoseabr/Projects/FluvialCO2/results32/"
input_vars = ['Por', 'Perm', 'gas_rate'] # Porosity, Permeability, ,  Well 'gas_rate', Pressure + x, y, time encodings 
output_vars = ['CO_2'] 



num_files= 100
traintest_split = 0.8

batch_size = 1

ntrain = num_files*traintest_split
ntest = num_files - ntrain

learning_rate = 0.001
epochs = 10 


iterations = epochs*(ntrain//batch_size)
modes = 12
width = 32 

################################################################
# configs-2
################################################################
# Prepare the path
path = 'fourier_3d_N{}_ep{}_m{}_w{}_b{}'.format(ntrain, epochs, modes, width, batch_size)

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



S = 32
#T_in = 61
T = 61

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

################################################################
# load data
################################################################
runtime = np.zeros(2, )
t1 = default_timer()
# Create instance of ReadXarrayDataset class for training data
dataset = ReadXarray(folder=folder, input_vars=input_vars, output_vars=output_vars, num_files = num_files, traintest_split = traintest_split)


# Get input and output data tensors
train_a = dataset.train_data_input #800, 32, 32, 61, 6 
train_u = dataset.train_data_output #800, 32, 32, 61, 1

test_a = dataset.test_data_input #200, 32, 32, 61, 6
test_u = dataset.test_data_output #200, 32, 32, 61, 1

# Move data tensors to GPU if available
train_a = train_a.to(device)
train_u = train_u.to(device)

test_a = test_a.to(device)
test_u = test_u.to(device)

# Normalize input_data and output_data
a_normalizer = UnitGaussianNormalizer(train_a)
#TODO: save a torch tensor with the mean and std of the training data

train_a= a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

y_normalizer = UnitGaussianNormalizer(train_u)
train_u = y_normalizer.encode(train_u)
test_u = y_normalizer.encode(test_u)

#save normalizers on path_model with the name of the path
torch.save(a_normalizer.mean, os.path.join(os.path.dirname(path_model), f'{path}_a_normalizer_mean.pt'))
torch.save(a_normalizer.std, os.path.join(os.path.dirname(path_model), f'{path}_a_normalizer_std.pt'))

torch.save(y_normalizer.mean, os.path.join(os.path.dirname(path_model), f'{path}_y_normalizer_mean.pt'))
torch.save(y_normalizer.std, os.path.join(os.path.dirname(path_model), f'{path}_y_normalizer_std.pt'))


t2 = default_timer()
#print shapes of normalized input and output data tensors
print("Train input data shape:", train_a.shape)
print("Train output data shape:", train_u.shape)
print("Test input data shape:", test_a.shape)
print("Test output data shape:", test_u.shape)
print('preprocessing finished, time used:', t2-t1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
#%%
################################################################
# training and evaluation
################################################################
model = FNO3d(modes, modes, modes, width).to(device) #TODO include .cuda()
print(count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)
y_normalizer #TODO include .cuda()
for ep in range(epochs):
    print(f'epoch {ep} of {epochs}')
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x.to(device) 
        y.to(device) 

        optimizer.zero_grad()
        out = model(x) #.view(batch_size, S, S, T)

        mse = F.mse_loss(out, y, reduction='mean')
        # mse.backward()

        y = y_normalizer.decode(y)
        out = y_normalizer.decode(out)
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
            x.to(device)
            y.to(device)

            out = model(x) #.view(batch_size, S, S, T)
            mse = F.mse_loss(out, y, reduction='mean')

            y = y_normalizer.decode(y)
            out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()            
            test_mse += mse.item()

            if index == 0:
                test_y_shape = (1, 61, 32, 32, 1)
                predicted_y_shape = (1, 61, 32, 32, 1)
                test_y = y[0].view(test_y_shape).cpu().numpy()
                predicted_y = out[0].view(predicted_y_shape).cpu().numpy()
                fig, ax = plt.subplots(nrows=1, ncols=3)
                ax[0].imshow(test_y[0, -1, :, :, 0].T)
                ax[1].imshow(predicted_y[0, -1, :, :, 0].T)
                ax[2].imshow((test_y[0, 0, :, :, 0]-predicted_y[0, 0, :, :, 0]).T)
                plt.savefig(path_image + '_ep' + str(ep) + '.png')
                plt.close()
       

    

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
pred = torch.zeros(test_u.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x.to(device) 
        y.to(device) 

        out = model(x)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        pred[index] = out

        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        print(index, test_l2)
        index = index + 1

scipy.io.savemat(os.path.join('runs', path, 'log', path+'.mat'), mdict={'pred': pred.cpu().numpy()})


###

