import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import os
from os import path
np.random.seed(20210218)
def lorenz_system(curr_state, t):
    """Lorenz System"""
    sigma = 10.
    rho = 28.
    beta = 8. / 3.

    x = curr_state[0]
    y = curr_state[1]
    z = curr_state[2]
    
    # define the 3 ODEs known as the lorenz equations
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    return [dx_dt, dy_dt, dz_dt]

def generate_trajectory(system, init_state, t_max, t_burn, dt):
    """runs dynamical system forward"""

    # generate time vector
    t = np.arange(0, t_max+t_burn, dt)
    t_start = round(t_burn/dt)
    trajectory = odeint(system, init_state, t)

    return t[t_start:], trajectory[t_start:,:]

def standardize_trajectory(traj, mean_traj=None, stddev_traj=None):

    if mean_traj is None:
        mean_traj = np.mean(traj, axis=0)
    if stddev_traj is None:
        stddev_traj = np.std(traj, axis=0)

    z_traj = np.divide((traj - mean_traj),stddev_traj)
    
    return z_traj

def minmax_norm_trajectory(traj):
    min_traj = np.min(traj, axis=0)
    max_traj = np.max(traj, axis=0)

    minmax_norm_traj = np.divide((traj - min_traj),(max_traj-min_traj))

    return minmax_norm_traj

def plot_traj(traj):
    fig = plt.figure();
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:,0], traj[:,1], traj[:,2])

    return ax

# repeated chops

runpath = '/snel/home/lwimala/tmp/deemg_tf2_test_runs/run_007/'
OVERWRITE = False
save_data = True
n_init_conds = 8
n_trials = 20
dt = 0.002 # seconds
t_max = 15 # seconds
t_burn = 3 # seconds, time to trim from beginning
data_dim = 12
lowd_dim = 3;
chop_len = 100 # bins
chop_olap = 20 # bins
tshift = 2 # bins
t_buffer = 2*tshift*dt # seconds, buffer for tshift
downsample_factor = 15
scale_norm = 1.0
bias = 1
# compute alpha/beta directly as projections from dynamical system
#w_alpha = np.random.rand(lowd_dim, data_dim) - 0.5
#w_beta = np.random.rand(lowd_dim, data_dim) - 0.5

#w_alpha_norm = np.linalg.norm(w_alpha)*scale_norm
#w_beta_norm = np.linalg.norm(w_beta)*scale_norm

#w_alpha = np.divide(w_alpha, w_alpha_norm)
#w_beta = np.divide(w_beta, w_beta_norm)

# compute transformations for mean from dynamical system and set variance to be mean dependent
w_mean = np.random.rand(lowd_dim, data_dim) - 0.5

w_mean_norm = np.linalg.norm(w_mean)*scale_norm
w_mean = np.divide(w_mean, w_mean_norm) # normalize projection matrix

# generate lowd trajectories from different initial conditions
all_lowd_traj = []
for i in range(n_init_conds):
    init_state = (np.random.rand(1,lowd_dim)-0.5)*20
    init_state = init_state.tolist()[0]
    print(init_state)
    t, lowd_traj = generate_trajectory(lorenz_system, init_state, t_max, t_burn, dt)

    traj_inds = np.arange(0, lowd_traj.shape[0], downsample_factor)

    lowd_traj = lowd_traj[traj_inds,:]
    t = t[traj_inds]

    all_lowd_traj.append(lowd_traj)

# compute mean and std from concat traj to standardize system 
concat_traj = np.vstack(all_lowd_traj)
mean_traj = np.mean(concat_traj,axis=0)
stddev_traj = np.std(concat_traj,axis=0)


# geenerate rates from lowd

all_rates = [] # time-varying mean
all_alpha = [] # Gamma dist. params
all_beta = []
all_lowd = []
for i in range(n_init_conds):
    #norm_lowd_traj = minmax_norm_trajectory(lowd_traj)

    #cent_lowd_traj = norm_lowd_traj - np.mean(norm_lowd_traj, axis=0)
    cent_lowd_traj = standardize_trajectory(all_lowd_traj[i], mean_traj=mean_traj, \
                                            stddev_traj=stddev_traj)

    log_mean = np.matmul(cent_lowd_traj, w_mean) 

    mean = np.exp(log_mean) + bias # bias prevents issues with values very close to 0

    var = 1.5 # fixed variance (can be adjusted to control SNR)

    # compute alpha and beta from mean/var relationships
    alpha = np.power(mean,2)/var
    beta = mean/var

    # if directly transforming to alpha and beta from underlying system 
    #log_alpha = np.matmul(cent_lowd_traj, w_alpha)
    #log_beta = np.matmul(cent_lowd_traj, w_beta)
    #alpha = np.exp(log_alpha)
    #beta = np.exp(log_beta)

    # rates 
    rates = mean
    all_alpha.append(alpha)
    all_beta.append(beta)
    all_rates.append(rates)
    all_lowd.append(cent_lowd_traj)

    if i==0:
        ax = plot_traj(cent_lowd_traj)
    else:
        ax.plot(cent_lowd_traj[:,0], cent_lowd_traj[:,1], cent_lowd_traj[:,2])


#all_beta = np.vstack(all_beta)
#all_alpha = np.vstack(all_beta) 
#data = [ np.random.gamma(alpha, 1/beta) for i in range(20)]
concat_alpha = np.vstack(all_alpha)
concat_beta = np.vstack(all_beta)
concat_rates = np.vstack(all_rates)
concat_lowd = np.vstack(all_lowd)

concat_data = [ np.random.gamma(concat_alpha, 1/concat_beta) for i in range(n_trials)]


# the below allows us to go from 
de = np.stack(concat_data)
n_samples, time_cond, dim = de.shape
concat_inds = np.arange(de.shape[1])
reshape_inds = np.reshape(concat_inds,[n_init_conds, t.size])

de2 = np.reshape(np.transpose(de, (0,2,1)),[n_samples, dim, n_init_conds, t.size ])

truth = np.reshape(concat_rates.T,[dim, n_init_conds, t.size])
truth_lowd = np.reshape(concat_lowd.T, [lowd_dim, n_init_conds, t.size])
#import pdb; pdb.set_trace()


if tshift > 0:
    tshift_chop_padding = np.ones((de2.shape[0],de2.shape[1],de2.shape[2], tshift))
    #tshift_chop_padding = np.ones((all_data.shape[0],tshift,all_data.shape[2]))*0.
    de2 = np.concatenate((tshift_chop_padding, de2,tshift_chop_padding), axis=3)
    #all_data = np.hstack([tshift_chop_padding, all_data, tshift_chop_padding])
    

n_chops = int(np.floor(t.size - chop_len) /(chop_len-chop_olap))

all_chops = []
all_truth_chops = []
all_truth_lowd_chops = []
all_chop_inds = []
for i in range(n_chops):
    start_idx = i*chop_len - i*chop_olap + tshift
    end_idx = start_idx + chop_len 
    chop = de2[:,:,:, start_idx-tshift:end_idx+tshift]
    truth_chop = truth[:,:,start_idx:end_idx]
    truth_lowd_chop = truth_lowd[:,:,start_idx:end_idx]
    truth_chops = np.reshape(np.tile(np.transpose(truth_chop, (0,2,1))[:,:,:,np.newaxis], (1,1,1,n_trials)), [dim, chop_len, n_init_conds*n_samples]).T
    truth_lowd_chops = np.reshape(np.tile(np.transpose(truth_lowd_chop, (0,2,1))[:,:,:,np.newaxis], (1,1,1,20)), [lowd_dim, chop_len, n_init_conds*n_samples]).T

    chop_inds = reshape_inds[:,np.arange(start_idx-tshift, end_idx-tshift)]
    chop_inds = np.reshape(np.transpose(np.tile(chop_inds[np.newaxis,:,:], (20,1,1)), (2,1,0)), [ chop_len, n_init_conds*n_samples]).T
    chops = np.reshape(np.transpose(chop,(1,3,2,0)), [dim, chop_len+(2*tshift), n_init_conds*n_samples]).T
    all_chops.append(chops)
    all_truth_chops.append(truth_chops)
    all_truth_lowd_chops.append(truth_lowd_chops)
    all_chop_inds.append(chop_inds)

true_lowd = np.vstack(all_truth_lowd_chops)
true = np.vstack(all_truth_chops)
chops = np.vstack(all_chops)
chop_inds = np.vstack(all_chop_inds)

print('Created %i chops!!' % chops.shape[0])

idx = np.random.permutation(chops.shape[0])
valid_inds = np.sort(idx[::5])
train_inds = np.sort(np.setdiff1d(idx,valid_inds))


datadir = path.join(runpath, 'deemg_input')
modeldir = path.join(runpath, 'deemg_output')

def check_exists(dirpath, overwrite=False):
    if path.exists(dirpath) and path.isdir(dirpath):
        print( 'WARNING: dir exists.' )
        if overwrite:
            import shutil
            print('INFO: Overwriting. Removing %s' % dirpath)
            shutil.rmtree(dirpath)

from collections import namedtuple

ChopParams = namedtuple(
    'ChopParams', [
        'chop_len',
        'chop_olap',
        'dim',
        'nconds',
        't',
        'ntrials_per_cond'
        ])
        

chop_params = ChopParams(
    chop_len=chop_len,
    chop_olap=chop_olap,
    dim=dim,
    nconds=n_init_conds,
    t=t.size,
    ntrials_per_cond=n_trials)

if save_data:
    check_exists(datadir, overwrite=OVERWRITE)
    check_exists(modeldir, overwrite=OVERWRITE)
    os.makedirs(datadir)
    os.makedirs(modeldir)
    filename = 'deemg_lorenz.h5'
    filepath = path.join(datadir, filename)
    with h5py.File(filepath, 'w') as h5f:
        h5f.create_dataset('train_data', data=chops[train_inds,:,:], dtype='float32', compression='gzip')
        h5f.create_dataset('train_true', data=true[train_inds,:,:], dtype='float32', compression='gzip')
        h5f.create_dataset('train_lowd', data=true_lowd[train_inds,:,:], dtype='float32', compression='gzip')        
        h5f.create_dataset('train_inds', data=train_inds, compression='gzip')
        h5f.create_dataset('train_idx', data=chop_inds[train_inds,:], compression='gzip')
    
        h5f.create_dataset('valid_data', data=chops[valid_inds,:,:], dtype='float32', compression='gzip')    
        h5f.create_dataset('valid_true', data=true[valid_inds,:,:], dtype='float32', compression='gzip')
        h5f.create_dataset('valid_lowd', data=true_lowd[valid_inds,:,:], dtype='float32', compression='gzip')        
        h5f.create_dataset('valid_inds', data=valid_inds, compression='gzip')
        h5f.create_dataset('valid_idx', data=chop_inds[valid_inds,:], compression='gzip')
        h5f.create_dataset('chop_params', data=chop_params, compression='gzip')    
        #if hasExtInputs:
        #    h5f.create_dataset('train_ext_input', data=np.transpose(train_ext), compression='gzip')
        #    h5f.create_dataset('valid_ext_input', data=np.transpose(valid_ext), compression='gzip')
        
    print('Sucessfully wrote the data to %s' % filepath)

import pdb; pdb.set_trace()
for i in range(chops.shape[2]):
    plt.figure();
    plt.plot(chops[:,:,i])
plt.show()

chan_idx = 4;
plt.figure()
plt.plot(data[0][:,chan_idx])
plt.plot(np.mean(data,axis=0)[:,chan_idx])
plt.plot(alpha[:,chan_idx]/beta[:,chan_idx])





import pdb; pdb.set_trace()
