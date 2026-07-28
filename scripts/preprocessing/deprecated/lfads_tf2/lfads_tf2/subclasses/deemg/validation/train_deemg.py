from lfads_tf2.utils import restrict_gpu_usage
restrict_gpu_usage(gpu_ix=0)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score
from os import path
import numpy as np
import h5py
import glob
from lfads_tf2.subclasses.deemg.models import deEMG
from lfads_tf2.utils import load_data, merge_chops, load_posterior_averages
from lfads_tf2.subclasses.deemg.defaults import get_cfg_defaults

if __name__=="__main__":
    cfg = get_cfg_defaults()
    cfg.TRAIN.DATA.DIR = '/snel/home/lwimala/Projects/future/wrist_iso/run_001/deemg_input/'
    cfg.TRAIN.DATA.PREFIX = 'deemg'
    cfg.TRAIN.MODEL_DIR = '/snel/home/lwimala/Projects/future/wrist_iso/run_001/deemg_output/'
    #cfg.TRAIN.MODEL_DIR = '/snel/home/lwimala/tmp/deemg_tf2_test_runs/pbt_run_004/best_model'
    cfg.TRAIN.OVERWRITE = True
    cfg.MODEL.DATA_DIM = cfg.MODEL.ENC_INPUT_DIM = 14
    cfg.MODEL.SEQ_LEN = 100
    cfg.MODEL.CO_DIM = 3
    cfg.MODEL.FAC_DIM = 10
    cfg.TRAIN.KL.IC_WEIGHT = 1e-3
    cfg.TRAIN.KL.CO_WEIGHT = 1e-3
    cfg.TRAIN.KL.INCREASE_EPOCH = 50
    cfg.TRAIN.L2.INCREASE_EPOCH = 50
    cfg.TRAIN.L2.GEN_SCALE = 1e-1
    cfg.TRAIN.L2.CON_SCALE = 1e-1
    cfg.TRAIN.DATA.AUGMENT.TEMPORAL_SHIFT = 6
    cfg.TRAIN.PATIENCE = 100 # number of epochs to wait before early stopping
    cfg.TRAIN.BATCH_SIZE = 400 # number of samples per batch
    cfg.TRAIN.LR.INIT = 0.002 # the initial learning rate
    cfg.TRAIN.EAGER_MODE = False
    cfg.TRAIN.MAX_EPOCHS = 100 # maximum number of training epochs
    cfg.MODEL.CD_RATE = 0.5
    cfg.freeze()

    # if training from scratch
    mode = 'training'
    # if sampling from trained model
    #mode = 'sampling'
    # if loading posterior averages
    #mode = 'loading'

    if mode == 'training':
        model = deEMG(cfg_node=cfg) # initialize from cfg node
        model.train() # train new model
        #model.sample_and_average()    
    elif mode == 'sampling':
        model_dir = cfg.TRAIN.MODEL_DIR
        model = deEMG(model_dir=model_dir) # load trained model
        model.sample_and_average()    
    elif mode == 'loading':
        model_dir = cfg.TRAIN.MODEL_DIR
    else:
        raise NotImplementedError('mode not recognized')
    exit(0)



emg  = load_data(cfg.TRAIN.DATA.DIR, \
                  prefix=cfg.TRAIN.DATA.PREFIX, signal='data', merge_tv=True)[0]
truth  = load_data(cfg.TRAIN.DATA.DIR, \
                  prefix=cfg.TRAIN.DATA.PREFIX, signal='true', merge_tv=True)[0]
lowd  = load_data(cfg.TRAIN.DATA.DIR, \
                  prefix=cfg.TRAIN.DATA.PREFIX, signal='lowd', merge_tv=True)[0]
idx  = load_data(cfg.TRAIN.DATA.DIR, \
                  prefix=cfg.TRAIN.DATA.PREFIX, signal='idx', merge_tv=True)[0]

with h5py.File(glob.glob(path.join(cfg.TRAIN.DATA.DIR, cfg.TRAIN.DATA.PREFIX + '*'))[0], 'r') as h5file:
    chop_len, chop_olap, dim, nconds, t, ntrials_per_cond = h5file['chop_params'][()]

deemg = load_posterior_averages(cfg.TRAIN.MODEL_DIR, merge_tv=True)[0]
# tempshift 6 model
cfg.defrost()
cfg.TRAIN.MODEL_DIR = '/snel/home/lwimala/tmp/deemg_tf2_test_runs/pbt_run_003/best_model'
cfg.freeze()
deemg_6 = load_posterior_averages(cfg.TRAIN.MODEL_DIR, merge_tv=True)[0]
tshift = cfg.TRAIN.DATA.AUGMENT.TEMPORAL_SHIFT


def merge_chops( chop_data, chop_len, chop_olap, dim=None ):

    w_left = np.arange(chop_olap)/chop_olap
    w_right = np.flip(w_left)

    wt = np.ones((chop_len))
    wt[:chop_olap] = w_left
    wt[-chop_olap:] = w_right

    if dim is None:
        dim = chop_data.shape[2]
    full_data =np.zeros((ntrials_per_cond,nconds, t, dim))
    
    for i in range(idx.shape[0]):
        trial_idx = np.mod(i,ntrials_per_cond)
        cond_idx = np.floor(idx[i,-1]/t).astype(np.int32)
        time_inds = np.mod(idx[i,:], t).astype(np.int32)

        # NOTE: Leftmost chop not handled properly to avoid applying weights to left edge 
        full_data[trial_idx,cond_idx,time_inds,:] += \
                                                     np.multiply(chop_data[i,:,:], \
                                                                 wt[:,np.newaxis])


    return full_data

deemg_full = merge_chops(deemg, chop_len, chop_olap)
deemg_6_full = merge_chops(deemg_6, chop_len, chop_olap)
emg_full = merge_chops(emg[:,tshift:-tshift,:], chop_len, chop_olap)
true_full = merge_chops(truth, chop_len, chop_olap)
lowd_full = merge_chops(lowd, chop_len, chop_olap)

nchops = int(np.floor(t - chop_len) /(chop_len-chop_olap))
end_idx = t - (chop_len + (chop_len-chop_olap)*(nchops-1))

deemg_6_full = deemg_6_full[:,:,chop_olap:-(end_idx+chop_olap)]
deemg_full = deemg_full[:,:,chop_olap:-(end_idx+chop_olap)]
emg_full = emg_full[:,:,chop_olap:-(end_idx+chop_olap)]
true_full = true_full[:,:,chop_olap:-(end_idx+chop_olap)]
lowd_full = lowd_full[:,:,chop_olap:-(end_idx+chop_olap)]



lowd_flat = lowd_full.transpose((1,0,2,3)).reshape(ntrials_per_cond*nconds*lowd_full.shape[2], lowd_full.shape[3])
deemg_flat = deemg_full.transpose((1,0,2,3)).reshape(ntrials_per_cond*nconds*lowd_full.shape[2], dim)
deemg_6_flat = deemg_6_full.transpose((1,0,2,3)).reshape(ntrials_per_cond*nconds*lowd_full.shape[2], dim)
true_flat = true_full.transpose((1,0,2,3)).reshape(ntrials_per_cond*nconds*lowd_full.shape[2], dim)

def compute_r2(y,yhat):
    ssr = np.sum(np.power(yhat-y,2),axis=0)
    sst = np.sum(np.power(y-np.mean(y,axis=0),2),axis=0)
    r2 = 1 - (ssr/sst)
    return r2

from sklearn.linear_model import LinearRegression

x = deemg_flat
x_6 = deemg_6_flat
y = lowd_flat
# fits intercept
lr = LinearRegression().fit(x,y) # object
lr_6 = LinearRegression().fit(x_6,y) # object

# get predictions
lowd_hat = lr.predict(deemg_flat)
lowd_6_hat = lr_6.predict(deemg_6_flat)

r2 = compute_r2(lowd_flat,lowd_hat)

print(r2)
r2_6 = compute_r2(lowd_flat,lowd_6_hat)
print(r2_6)
fig = plt.figure();
ax = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

# for each condition
import matplotlib.cm as colormap

cm = colormap.hsv
for i in range(nconds):
    lowd_plot = lowd_full[0,i,:,:]
    lowd_pred_plot = lr.predict(np.mean(deemg_full[:,i,:,:],axis=0))
    lowd_6_pred_plot = lr_6.predict(np.mean(deemg_6_full[:,i,:,:],axis=0))
    ax.plot(lowd_plot[:,0], lowd_plot[:,1], lowd_plot[:,2], alpha=0.3, color=cm(float(i/nconds)), linewidth=1 )
    ax2.plot(lowd_pred_plot[:,0], lowd_pred_plot[:,1], lowd_pred_plot[:,2], alpha=0.3, color=cm(float(i/nconds)), linewidth=1 )
    ax3.plot(lowd_6_pred_plot[:,0], lowd_6_pred_plot[:,1], lowd_6_pred_plot[:,2], alpha=0.3, color=cm(float(i/nconds)), linewidth=1 )
ax.set_title('True Latent')
from textwrap import wrap
ax2.set_title("\n".join(wrap('Affine Transform of deEMG tshift=2',20)))
ax3.set_title("\n".join(wrap('Affine Transform of deEMG tshift=6',20)))
plt.suptitle('Fit R^2 tshift=2: ' + str(r2) + '\nFit R^2 tshift=6: ' + str(r2_6))

plt.figure();
plt.plot(range(1,12), [compute_r2(true_flat[:-i,:], deemg_flat[i:,:]) for i in range(1,12)], color='r', label='tshift=2')
plt.plot(range(1,12), [compute_r2(true_flat[:-i,:], deemg_6_flat[i:,:]) for i in range(1,12)], color='b', label='tshift=6')
plt.xlabel('Lag (# of bins)')
plt.ylabel('Mean R^2 acrosss channels')
plt.legend()

chans = [ 1, 4 ]
conds = [ 0, 1 ]

fig = plt.figure()
count = 1;
opacity = 0.2;
for i, ichan in enumerate(chans):
    for j, icond in enumerate(conds):
        ax = fig.add_subplot(len(chans),len(conds),count)
        ax.plot(true_full[0,icond,:,ichan], color='k')
        ax.plot(deemg_full[:,icond,:,ichan].T,color='r', alpha=opacity)
        count += 1
        if i==0:
            ax.set_title('Condition ' + str(j+1))
        if j==0:
            ax.set_ylabel('Channel ' + str(ichan))
plt.suptitle('tshift=2 model single trial rates')
fig = plt.figure()
count = 1;
for i, ichan in enumerate(chans):
    for j, icond in enumerate(conds):
        ax = fig.add_subplot(len(chans),len(conds),count)
        ax.plot(true_full[0,icond,:,ichan], color='k')
        ax.plot(deemg_6_full[:,icond,:,ichan].T,color='b', alpha=opacity)
        count += 1
        if i==0:
            ax.set_title('Condition ' + str(j+1))
        if j==0:
            ax.set_ylabel('Channel ' + str(ichan))
plt.suptitle('tshift=6 model single trial rates')
print('check plot')

fig = plt.figure()
ichan = 3
icond = 0
ax = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)


count = 1;
opacity = 0.6;

ax.plot(true_full[0,icond,:,ichan], color='k')
ax.plot(np.mean(deemg_full[:,icond,:,ichan],axis=0),color='r', alpha=opacity)
ax.plot(np.mean(deemg_6_full[:,icond,:,ichan],axis=0),color='b', alpha=opacity)

ax2.plot(true_full[0,icond,:,ichan], color='k')
ax2.plot(np.mean(deemg_full[2:,icond,:,ichan],axis=0),color='r', alpha=opacity)
ax2.plot(np.mean(deemg_6_full[6:,icond,:,ichan],axis=0),color='b', alpha=opacity)
ax.set_title            
print('check plot')


plt.show()
import pdb; pdb.set_trace()
