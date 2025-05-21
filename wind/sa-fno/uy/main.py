# This script is for the response prediction on a 37th storey building under wind loading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import os
import time
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
import sys
from Adam import Adam
import scipy
from scipy.io import loadmat, savemat
torch.manual_seed(0)
np.random.seed(0)


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        #x_ft = torch.rfft(x,1,normalized=True,onesided=False)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(sstorey+1, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
    

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0.2*(cutoff-1), 600, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

class PARAM(nn.Module):
    def __init__(self, dim):
        super(PARAM, self).__init__()
        self.lam = nn.Parameter(torch.ones(dim, 1))  
        def forward(self):
            pass

def exp_mask(lam):
    return torch.exp(lam)

def clip_weights(model_param):
    with torch.no_grad():
        model_param.lam.data = torch.clamp(model_param.lam.data, min=-5, max=5)
################################################################
#  configurations
################################################################
# device = torch.device('cuda')
# Recording the response of the highest floor
          
save_index = 1
batch_size_train = 200
batch_size_test = 200
learning_rate = 0.001

epochs = 300
step_size = 50
gamma = 0.5

modes = 450
width = 64

num_train = 800
num_test = 200

################################################################
# load data and data normalization
################################################################
cutoff = 100
rtime = 30
num_storeys = 37
rstoreys = 1

case = "Results"

index = loadmat('../../data/train_test_index.mat')
train_index = index['train'][0,:].T
test_index = index['test'][0,:].T

reader = MatReader('../../data/windData1.mat')
x1 = reader.read_field('f')[:,0:-cutoff][:,::rtime, ::rstoreys]
y1 = reader.read_field('uy')[:,cutoff::][:,::rtime, -1]

reader = MatReader('../../data/windData2.mat')
x2 = reader.read_field('f')[:,0:-cutoff][:,::rtime, ::rstoreys]
y2 = reader.read_field('uy')[:,cutoff::][:,::rtime, -1]

reader = MatReader('../../data/windData3.mat')
x3 = reader.read_field('f')[:,0:-cutoff][:,::rtime, ::rstoreys]
y3 = reader.read_field('uy')[:,cutoff::][:,::rtime, -1]

reader = MatReader('../../data/windData4.mat')
x4 = reader.read_field('f')[:,0:-cutoff][:,::rtime, ::rstoreys]
y4 = reader.read_field('uy')[:,cutoff::][:,::rtime, -1]

reader = MatReader('../../data/windData5.mat')
x5 = reader.read_field('f')[:,0:-cutoff][:,::rtime, ::rstoreys]
y5 = reader.read_field('uy')[:,cutoff::][:,::rtime, -1]

reader = MatReader('../../data/windData6.mat')
x6 = reader.read_field('f')[:,0:-cutoff][:,::rtime, ::rstoreys]
y6 = reader.read_field('uy')[:,cutoff::][:,::rtime, -1]

x = torch.cat((x1, x2, x3, x4, x5, x6), axis =0)
y = torch.cat((y1, y2, y3, y4, y5, y6), axis =0)

x_train = x[train_index, :, :]
y1_train = y[train_index, :,]
x_test = x[test_index,:,:]
y1_test = y[test_index,:]

stime = x_train.shape[1]
sstorey = x_train.shape[-1]
ntrain = y1_train.shape[0]
ntest = y1_test.shape[0]
print(x_train.shape)

print("Save Index = ", save_index)
print("Number of modes = ", modes)
print("Rate of downsampling in time = ", rtime)
print("Total number of time points = ", stime)
print("Rate of downsampling in storeys  = ", rstoreys)
print("Total number of storeys = ", sstorey)
print("Number of time points delayed = ", cutoff)

x_train = x_train.reshape(ntrain,stime,sstorey)
x_test = x_test.reshape(ntest,stime,sstorey)

y1_train = y1_train.reshape(ntrain,stime,1)
y1_test = y1_test.reshape(ntest,stime,1)

x_normalizer = GaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y1_normalizer = GaussianNormalizer(y1_train)
y1_train = y1_normalizer.encode(y1_train)
y1_test = y1_normalizer.encode(y1_test)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y1_train), batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y1_test), batch_size=batch_size_test, shuffle=False)

# model
model = FNO1d(modes, width)#.cuda()
model_param = PARAM(stime)
print("Number of trainable parameters = ", count_params(model))
sub_model_params = [{
    "params": model_param.parameters(), 
    "weight_decay": 1e-6, 
    "lr": 1e-2}]

################################################################
# training and evaluation
################################################################
optimizer1 = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
optimizer2 = torch.optim.Adam(sub_model_params)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=step_size, gamma=gamma)
scheduler_lambda = torch.optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.95)
start_time = time.time()
myloss = LpLoss(size_average=True)

train_error = np.zeros((epochs, 1))
train_loss = np.zeros((epochs, 1))
test_error = np.zeros((epochs, 1))
test_loss = np.zeros((epochs, 1))

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y1 = x, y

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        out1 = model(x)
        lamb = model_param.lam
        mask = exp_mask(lamb)
        reg_term = 0.001 * torch.mean(mask)
        loss = torch.sum(torch.mean(torch.einsum('ijk,jk->ijk', torch.square(out1 - y1), mask), dim=0)) + reg_term
        #loss = torch.sum(torch.mean(torch.einsum('ijk,jk->ijk', torch.square(out1 - y1),lamb**2), dim=0))
        loss.backward()
        y1 = y1_normalizer.decode(y1)
        out1 = y1_normalizer.decode(out1)
        l2 = myloss(out1, y1)

        optimizer1.step()
        optimizer2.step()
        train_mse += loss.item()
        train_l2 += l2.item()
        clip_weights(model_param)

    scheduler.step()
    scheduler_lambda.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y1 = x, y #x.cuda(), y.cuda()

            out1 = model(x)
            
            y1 = y1_normalizer.decode(y1)
            out1 = y1_normalizer.decode(out1)            
            test_l2 += myloss(out1, y1).item()

    train_mse /= len(train_loader)
    train_l2 /= num_train
    test_l2 /= num_test

    train_loss[ep,0] = train_mse
    train_error[ep,0] = train_l2
    test_error[ep,0] = test_l2
    t2 = default_timer()
    print("Epoch: %d, time: %.3f, Train Loss: %.3e, Train l2: %.4f, Test l2: %.4f" % (ep, t2-t1, train_mse, train_l2, test_l2))

elapsed = time.time() - start_time
print("\n=============================")
print("Training done...")
print('Training time: %.3f'%(elapsed))
print("=============================\n")

# ====================================
# saving settings
# ====================================
current_directory = os.getcwd()
folder_index = str(save_index)

results_dir = "/" + case + "/"
save_results_to = current_directory + results_dir
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

x = np.linspace(0, epochs-1, epochs)
np.savetxt(save_results_to+'/epoch.txt', x)
np.savetxt(save_results_to+'/train_loss.txt', train_loss)
np.savetxt(save_results_to+'/test_loss.txt', test_loss)
np.savetxt(save_results_to+'/train_error.txt', train_error)
np.savetxt(save_results_to+'/test_error.txt', test_error)    
save_models_to = save_results_to +"model/"
if not os.path.exists(save_models_to):
    os.makedirs(save_models_to)
    
torch.save(model, save_models_to+'WindResponse')

################################################################
# testing
################################################################
batch_size_test = 1
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y1_test), batch_size=batch_size_test, shuffle=False)    
pred_u1 = torch.zeros(num_test,stime,1)

index = 0
test_l2 = 0
t1 = default_timer()
dataSegment = "Test"
with torch.no_grad():
    for x, y in test_loader:
        
        x, y1 = x, y
        out1 = model(x)
        y1 = y1_normalizer.decode(y1)
        out1 = y1_normalizer.decode(out1)            
        pred_u1[index,:,:] = out1    
        test_l2 += myloss(out1, y1).item()       
        index = index + 1

test_l2 = test_l2/index
t2 = default_timer()
testing_time = t2-t1

x_test = x_normalizer.decode(x_test)
y1_test = y1_normalizer.decode(y1_test)
scipy.io.savemat(save_results_to+'WindResponse_test.mat', 
                  mdict={'x_test': x_test.detach().cpu().numpy(),
                         'y1_test': y1_test.numpy(), 
                         'y1_pred': pred_u1.cpu().numpy(),
                         'train_l2': test_l2})  

print("\n=============================")
print('Testing error: %.3e'%(test_l2))
print("=============================\n")    

# Plotting the loss history
num_epoch = epochs
x = np.linspace(1, num_epoch, num_epoch)
fig = plt.figure(constrained_layout=False, figsize=(7, 7))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0])
ax.plot(x, train_loss[:,0], color='blue', label='Training Loss')
ax.plot(x, test_loss[:,0], color='red', label='Testing Loss')
ax.set_yscale('log')
ax.set_ylabel('Loss')
ax.set_xlabel('Epochs')
ax.legend(loc='upper left')
fig.savefig(save_results_to+'loss_history.png')
