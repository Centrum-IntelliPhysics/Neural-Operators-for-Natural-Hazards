import jax.numpy as jnp
import numpy.random as npr
from jax import jit, grad, vmap
from jax.example_libraries.optimizers import adam
from jax import value_and_grad
from functools import partial
from jax import jacfwd, jacrev
import jax.nn as jnn
import math
from jax import random
import jax
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from flax import linen as nn
import sklearn.metrics
from jax.lax import conv_general_dilated as conv_lax

import argparse
import os
import time
from termcolor import colored
from scipy.io import loadmat
import scipy.io as io
import pickle

import sys
sys.path.append("../..")

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9})
import seaborn as sns
sns.set_style("white")
sns.set_style("ticks")

import warnings
warnings.filterwarnings("ignore")

# Check where gpu is enable or not
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

cluster = False
save = True

if cluster == True:
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', dest='seed', type=int, default=0, help='Seed number.')
    args = parser.parse_args()

    # Print all the arguments
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    seed = args.seed

if cluster == False:
    seed = 0 # Seed number.

if save == True:
    resultdir = os.path.join(os.getcwd(), 'Results')
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

if save == True and cluster == True:
    orig_stdout = sys.stdout
    q = open(os.path.join(resultdir, 'outputs.txt'), 'w')
    sys.stdout = q
    print ("------START------")

print('seed = '+str(seed))
np.random.seed(seed)
key = 1234 #random.PRNGKey(seed)

# Load the data
index = loadmat('../../data/train_test_index.mat')
train_index = index['train'][0,:].T
test_index = index['test'][0,:].T

cutoff = 100
rtime = 30
num_storeys = 37
rstoreys = 1

# Load the data
reader = loadmat('../../data/windData1.mat')
x1 = reader['f'][:,0:-cutoff][:,::rtime, ::rstoreys]
y1 = reader['uz'][:,cutoff::][:,::rtime, -1]

reader = loadmat('../../data/windData2.mat')
x2 = reader['f'][:,0:-cutoff][:,::rtime, ::rstoreys]
y2 = reader['uz'][:,cutoff::][:,::rtime, -1]

reader = loadmat('../../data/windData3.mat')
x3 = reader['f'][:,0:-cutoff][:,::rtime, ::rstoreys]
y3 = reader['uz'][:,cutoff::][:,::rtime, -1]

reader = loadmat('../../data/windData4.mat')
x4 = reader['f'][:,0:-cutoff][:,::rtime, ::rstoreys]
y4 = reader['uz'][:,cutoff::][:,::rtime, -1]

reader = loadmat('../../data/windData5.mat')
x5 = reader['f'][:,0:-cutoff][:,::rtime, ::rstoreys]
y5 = reader['uz'][:,cutoff::][:,::rtime, -1]

reader = loadmat('../../data/windData6.mat')
x6 = reader['f'][:,0:-cutoff][:,::rtime, ::rstoreys]
y6 = reader['uz'][:,cutoff::][:,::rtime, -1]

x = np.concat((x1, x2, x3, x4, x5, x6), axis = 0)
y = np.concat((y1, y2, y3, y4, y5, y6), axis = 0)
t = np.array(np.linspace(0,1,997)).reshape(997,1)

inputs_train = jnp.array(x[train_index])
outputs_train = jnp.array(y[train_index])
inputs_test = jnp.array(x[test_index])
outputs_test = jnp.array(y[test_index])
grid = jnp.array(t)
#print("grid:", grid)

# Check the shapes of the subsets
print("Shape of inputs_train:", inputs_train.shape)
print("Shape of inputs_test:", inputs_test.shape)
print("Shape of outputs_train:", outputs_train.shape)
print("Shape of outputs_test:", outputs_test.shape)
print('#'*100)

# Scaling the inputs
inputs_mean = jnp.mean(inputs_train, axis = 0)
input_std = jnp.std(inputs_train, axis = 0)
outputs_mean = jnp.mean(outputs_train, axis = 0)
outputs_std = jnp.std(outputs_train, axis = 0)

inputs_train = (inputs_train - inputs_mean)/input_std
outputs_train = (outputs_train - outputs_mean)/outputs_std
inputs_test = (inputs_test - inputs_mean)/input_std
outputs_test = (outputs_test - outputs_mean)/outputs_std

# Initialize the Glorot (Xavier) normal distribution for weight initialization
initializer = jax.nn.initializers.glorot_normal()
rng_key = random.PRNGKey(0)
key1, key2, key3, key4 = random.split(rng_key, 4)

def conv(x, w, b):
    """Convolution operation with VALID padding"""
    # Reshape inputs to match JAX's conv_general_dilated expectations
    # x shape: (H, W, C_in)
    # w shape: (H, W, C_in, C_out)
    conv_out = conv_lax(
        lhs=x[None, ...],  # Add batch dimension: (1, H, W, C_in)
        rhs=w,             # Kernel: (H, W, C_in, C_out)
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    return conv_out[0] + b  # Remove batch dimension and add bias

def init_cnn_params(p, key=random.PRNGKey(0)):
    """
    Initialize CNN parameters for the branch network as a list of tuples.
    Each tuple contains (weights, biases) for a layer.

    Returns:
    list: List of tuples, each containing weights and biases for a layer
    """
    key1, key2, key3, key4, key5 = random.split(key, 5)

    # Conv1: (3,3,1,32) - input has 1 channel, 32 output filters
    conv1_w = random.normal(key1, (8, 8, 1, 6)) * 0.1
    conv1_b = random.normal(key2, (6,)) * 0.1

    conv2_w = random.normal(key2, (8, 8, 6, 6)) * 0.1
    conv2_b = random.normal(key3, (1,)) * 0.1

    conv3_w = random.normal(key3, (8, 8, 6, 6)) * 0.1
    conv3_b = random.normal(key4, (6,)) * 0.1

    flat_size = 93696 #TODO: change this to automatic

    dense1_w = random.normal(key4, (flat_size, 256)) * jnp.sqrt(2.0 / flat_size)
    dense1_b = jnp.zeros(256)

    dense2_w = random.normal(key5, (256, p)) * jnp.sqrt(2.0 / 256)
    dense2_b = jnp.zeros(p)

    return [
        (conv1_w, conv1_b),
        (conv2_w, conv2_b),
        (conv3_w, conv3_b),
        (dense1_w, dense1_b),
        (dense2_w, dense2_b)
    ]

def BranchNet(params, x):
    """
    CNN-based branch network for the DeepONet.

    Args:
    params (list): List of tuples containing weights and biases
    x (array): Input tensor of shape (batch_size, 41, 41)

    Returns:
    array: Output tensor of shape (batch_size, p)
    """
    def single_forward(params, x):
        # Unpack conv and dense layer parameters
        (conv1_w, conv1_b), (conv2_w, conv2_b), (conv3_w, conv3_b), \
        (dense1_w, dense1_b), (dense2_w, dense2_b) = params

        # Reshape input to (997, 37, 1) - adding channel dimension
        x = x.reshape(997, 37, 1)

        # Convolution layers with SiLU activation
        x = jnn.silu(conv(x, conv1_w, conv1_b))
        x = jnn.silu(conv(x, conv2_w, conv2_b))
        x = jnn.silu(conv(x, conv3_w, conv3_b))

        # Flatten
        x = x.reshape(-1)

        # Dense layers
        x = jnn.silu(jnp.dot(x, dense1_w) + dense1_b)
        outputs = jnp.dot(x, dense2_w) + dense2_b

        return outputs

    return vmap(partial(single_forward, params))(x)

def init_glorot_params(layer_sizes, key = random.PRNGKey(seed)):
    """
    Initialize the parameters of the neural network using Glorot (Xavier) initialization.

    Args:
    layer_sizes (list): List of integers representing the size of each layer.
    key (PRNGKey): Random number generator key for reproducibility.

    Returns:
    list: List of tuples, each containing weights and biases for a layer.
    """
    return [(initializer(key, (m, n), jnp.float32), jnp.zeros(n))
            for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

def TrunkNet(params, x):
    """
    Implement the trunk network of the DeepONet.

    Args:
    params (list): List of weight and bias tuples for each layer.
    x (float): First input to the trunk network.
    t (float): Second input to the trunk network.

    Returns:
    array: Output of the trunk network.
    """
    inputs = jnp.array(x)
    for w, b in params:
        outputs = jnp.dot(x, w) + b
        x = jnn.silu(outputs)
    return outputs

@jit
def DeepONet(params, branch_inputs, trunk_inputs):
    """
    Implement the complete DeepONet architecture.

    Args:
    params (tuple): Tuple containing branch and trunk network parameters.
    branch_inputs (array): Inputs for the branch network.
    trunk_inputs (array): Inputs for the trunk network.

    Returns:
    array: Output of the DeepONet.
    """
    params_branch, params_trunk = params
    branch_outputs = lambda x: BranchNet(params_branch, x)
    b_out = branch_outputs(branch_inputs)
    #b_out = b_out.reshape(-1,num_storeys,p//num_storeys)
    trunk_output = lambda y: TrunkNet(params_trunk, y)
    t_out = trunk_output(trunk_inputs)
    #t_out = t_out.reshape(-1,num_storeys,p//num_storeys)
    results = jnp.einsum('ik, lk -> il',b_out, t_out)
    return results

# network parameters.
p = 300 # Number of output neurons in both the branch and trunk net outputs.
nx = 101
input_neurons_branch = nx # m
input_neurons_trunk = 1

layer_sizes_t = [input_neurons_trunk] + [100]*6 + [p]

params_branch = init_cnn_params(p)
params_trunk = init_glorot_params(layer_sizes=layer_sizes_t)

params= (params_branch, params_trunk)

def objective(params, branch_inputs, trunk_inputs, target_values):
    """
    Define the objective function (loss function) for training.

    Args:
    params (tuple): Tuple containing branch and trunk network parameters.
    branch_inputs (array): Inputs for the branch network.
    trunk_inputs (array): Inputs for the trunk network.
    target_values (array): True output values to compare against.

    Returns:
    float: Mean squared error loss.
    """
    predictions = DeepONet(params, branch_inputs, trunk_inputs)
    loss_mse = jnp.mean((predictions - target_values)**2)
    return loss_mse


# Adam optimizer
@jit
def update(params, branch_input, trunk_inputs, target_values, opt_state):
    """
    Compute the gradient for a batch and update the parameters.

    Args:
    params (tuple): Current network parameters.
    branch_inputs (array): Inputs for the branch network.
    trunk_inputs (array): Inputs for the trunk network.
    target_values (array): True output values.
    opt_state: Current state of the optimizer.

    Returns:
    tuple: Updated parameters, updated optimizer state, and current loss value.
    """
    value, grads = value_and_grad(objective)(params, branch_input, trunk_inputs, target_values)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value

# Initialize the Adam optimizer
opt_init, opt_update, get_params = adam(step_size=1e-3, b1=0.9, b2=0.999, eps=1e-08)
opt_state = opt_init(params)

bs = 500 #batch size
iteration_list, loss_list, test_loss_list = [], [], []
iteration = 0

n_epochs = 100000
num_samples = len(inputs_train)

# test input preparation
branch_inputs_test = inputs_test
targets = outputs_test

def save_model_params(params, resultdir, filename='model_params_deeponet.pkl'):
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    save_path = os.path.join(resultdir, filename)
    with open(save_path, 'wb') as f:
        pickle.dump(params, f)

def load_model_params(resultdir, filename='model_params_deeponet.pkl'):
    load_path = os.path.join(resultdir, filename)
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return params

# Saving
if save:
    save_model_params(params, resultdir)

# Loading (uncomment when needed)
# model_params = load_model_params(resultdir)

## Training of DeepONet
start = time.time() # start time of training
best_test_mse = float('inf')  # Initialize with infinity

# Save initial model at 0th iteration
save_model_params(params, resultdir, filename='model_params_best_deeponet.pkl')
print("Saved initial model at iteration 0")

for iteration in range(n_epochs):
    indices = jax.random.permutation(jax.random.PRNGKey(0), num_samples)
    batch_index = indices[0:bs]
    inputs_train_shuffled = inputs_train[batch_index]
    outputs_train_shuffled = outputs_train[batch_index]
    target_values = outputs_train_shuffled
    branch_inputs = inputs_train_shuffled
    trunk_inputs = grid
    params, opt_state, value = update(params, branch_inputs, trunk_inputs, target_values, opt_state)

    if iteration % 1000 == 0:
        params_branch, params_trunk = params
        predictions = DeepONet(params, branch_inputs, trunk_inputs)
        test_mse = jnp.mean((predictions - target_values)**2)

        # Compare current test error with the best so far
        if test_mse < best_test_mse:
            best_test_mse = test_mse
            # Save the model as it's the best so far
            save_model_params(params, resultdir, filename='model_params_best_deeponet.pkl')
            print(f"New best model saved at iteration {iteration} with test MSE: {test_mse:.7f}")

        finish = time.time() - start
        print(f"Iteration: {iteration:3d}, Train loss: {objective(params, branch_inputs, trunk_inputs, target_values):.7f}, Test loss: {test_mse:.7f}, Best test loss: {best_test_mse:.7f}, Time: {finish:.2f}")

    iteration_list.append(iteration)
    loss_list.append(objective(params, branch_inputs, trunk_inputs, target_values))
    test_loss_list.append(test_mse)

if save:
    np.save(os.path.join(resultdir, 'iteration_list_deeponet.npy'), np.asarray(iteration_list))
    np.save(os.path.join(resultdir, 'loss_list_deeponet.npy'), np.asarray(loss_list))
    np.save(os.path.join(resultdir, 'test_loss_list_deeponet.npy'), np.asarray(test_loss_list))

# Plotting code remains the same
plt.figure()
plt.plot(iteration_list, loss_list, 'g', label='Training loss')
plt.plot(iteration_list, test_loss_list, '-b', label='Test loss')
plt.yscale("log")
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

if save:
    plt.savefig(os.path.join(resultdir, 'loss_plot_deeponet.pdf'))

# end timer
finish = time.time() - start
print("Time (sec) to complete:\n" + str(finish))

# params_branch, params_trunk = params
# Load the best model parameters
best_params = load_model_params(resultdir, filename='model_params_best_deeponet.pkl')
print("Loaded best model parameters")

# Predictions
mse_list = []

branch_inputs = inputs_test
trunk_inputs = grid
prediction = DeepONet(best_params, branch_inputs, trunk_inputs) # (bs, neval)

inputs_save = inputs_test*input_std + inputs_mean
outputs_save = outputs_test*outputs_std + outputs_mean
prediction_save = prediction*outputs_std + outputs_mean

save_dict = {'ground_motion': inputs_save, 'disp_pred': prediction_save,\
             'disp_target': outputs_save, 'grid': grid}

io.savemat(resultdir+'/pred_deeponet_test.mat', save_dict)
del prediction, inputs_save, outputs_save, prediction_save

branch_inputs = inputs_train
trunk_inputs = grid
prediction = DeepONet(best_params, branch_inputs, trunk_inputs) # (bs, neval)

inputs_save = inputs_train*input_std + inputs_mean
outputs_save = outputs_train*outputs_std + outputs_mean
prediction_save = prediction*outputs_std + outputs_mean

save_dict = {'ground_motion': inputs_save, 'disp_pred': prediction_save,\
             'disp_target': outputs_save, 'grid': grid}

io.savemat(resultdir+'/pred_deeponet_train.mat', save_dict)

for i in range(inputs_test.shape[0]):

    branch_inputs = inputs_test[i].reshape(1, inputs_test[i].shape[0], inputs_test[i].shape[1])
    trunk_inputs = grid # (neval, 1)

    prediction_i = DeepONet(best_params, branch_inputs, trunk_inputs) # (bs, neval)
    target_i = outputs_test[i]

    prediction_i = prediction_i*outputs_std + outputs_mean
    target_i = target_i*outputs_std + outputs_mean

    mse_i = np.mean((prediction_i - target_i)**2)
    mse_list.append(mse_i.item())

    if i % 1000 == 0:
        print(colored('TEST SAMPLE '+str(i+1), 'red'))

        r2score = metrics.r2_score(target_i.flatten(), prediction_i.flatten())
        relerror = np.linalg.norm(target_i- prediction_i) / np.linalg.norm(target_i)
        r2score = float('%.4f'%r2score)
        relerror = float('%.4f'%relerror)
        print('Rel. L2 Error = '+str(relerror)+', R2 score = '+str(r2score))

        fig = plt.figure(figsize=(15, 4))

        # Adjust subplot parameters for better spacing
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.3)

        # Input plot
        ax = fig.add_subplot(1, 3, 1)
        inputs_print = inputs_test[i]*input_std + inputs_mean
        plt.plot(inputs_print[:,-1])
        plt.title('Input', fontsize=14)

        # Output plot
        ax = fig.add_subplot(1, 3, 2)
        target = target_i.reshape(grid.shape[0])
        prediction = prediction_i.reshape(grid.shape[0])
        plt.plot(target, color='blue', linewidth=2)
        plt.plot(prediction, color='red', linewidth=2)
        plt.title('Output Field', fontsize=14)
        plt.legend(['Target', 'Prediction'])

        # Error plot
        ax = fig.add_subplot(1, 3, 3)
        error = target - prediction
        plt.plot(error, color='magenta')
        plt.yscale("log")
        plt.title('Absolute Error', fontsize=14)

        print(colored('#'*230, 'green'))

mse = sum(mse_list) / len(mse_list)
print("Mean Squared Error Test :\n", mse)

