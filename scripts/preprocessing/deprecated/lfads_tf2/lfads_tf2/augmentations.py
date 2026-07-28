"""Functions for augmenting neural spiking data.

This module contains functions that modify spiking data immediately
prior to passing it into the model. They are best used in combination
with the `tf.Dataset.map` function.
"""

import numpy as np 
import tensorflow as tf 
from lfads_tf2.tuples import BatchInput

class Jitter:
    def __init__(self):
        # define augment func
        self.augment_func = self.shuffle_spikes_in_time
        
    def shuffle_spikes_in_time(self, batch_data, cfg_node): 
        """
        A wrapper function around numpy_jitter that allows the numpy function to 
        be applied to tensors. Handles the application of jitter to a portion of 
        the BatchInput tuple via tf.map in models.py.

        Parameters
        ----------
        batch_data : lfads_tf2.tuples.BatchInput
            BatchInput object containing spike count data to be 
            shuffled as well as external inputs and sv_mask 
        cfg_node : yacs node
            Sub-node of main config cfg.TRAIN.DATA.AUGMENT

        Returns
        -------
        lfads_tf2.tuples.BatchInput
            BatchInput object containing shuffled spike data and 
            the unchanged external inputs and sv_mask.
        """

        # if jitter width is 0, then return the data unchanged
        w = cfg_node.JITTER_WIDTH
        if w == 0:
            return batch_data
        # define the jitter function
        jitter = lambda data: numpy_jitter(data, w)
        # turn the numpy impl into a tf function and apply to data
        shuffled_spikes = tf.numpy_function(jitter, [batch_data.data], tf.float32)
        
        # return a new BatchInput tuple with shuffled spikes
        return BatchInput(
            data=shuffled_spikes,
            recon_data=shuffled_spikes, 
            sv_mask=batch_data.sv_mask,
            recon_sv_mask=batch_data.sv_mask, 
            ext_input=batch_data.ext_input,
            dataset_id_ixs=batch_data.dataset_id_ixs)

class Shift:
    def __init__(self):
        # define augment func
        self.augment_func = self.apply_temporal_shift_v2

    def apply_temporal_shift_v2(self, batch_data, cfg_node):
        """
        v2: aims to remove need for padding the data before modeling. Will pad
        dataset with buffers of length tshift using nan filling. Then will shift
        channels. nan's will be blocked by selective backprop.

        Shifts channels randomly in the temporal dimension.  This is useful to
        help the LFADS system avoid overfitting to individual spikes or fast
        oscillations found in the data that are irrelevant to behavior. A
        pure 'tabula rasa' approach would avoid this, but LFADS is sensitive
        enough to pick up dynamics that you may not want.

        Parameters
        ----------
        batch_data : lfads_tf2.tuples.BatchInput
            BatchInput object containing data to be 
            shifted as well as external inputs and sv_masks 
        cfg_node : yacs node
            Sub-node of main config cfg.TRAIN.DATA.AUGMENT
        tshift : float
            the maximum number of bins to shift channels in either 
            direction
        shift_dist : string
            specifies the type of random distribution to sample shifts
            default is a truncated normal distribution
        """

        tshift = cfg_node.TEMPORAL_SHIFT
        shift_dist = cfg_node.TEMPORAL_SHIFT_DIST

        if tshift == 0:
            return batch_data
        
        # data/mask used for input to encoders
        data = batch_data.data
        sv_mask = batch_data.sv_mask 

        seq_len, d = data.shape
        
        paddings = tf.constant([[tshift,tshift],[0,0]])
        pad_data = tf.pad(data, paddings,
                                mode="CONSTANT", constant_values=np.nan)

        pad_sv_mask = tf.pad(sv_mask, paddings,
                                mode="CONSTANT", constant_values=0)
        
        
        shifts = generate_random_shift(d*2, tshift, shift_dist)
        input_shift, recon_shift = tf.split(shifts, num_or_size_splits=2)
        
        input_data = tf.stack([pad_data[input_shift[i]:input_shift[i]+seq_len, i] for i in range(d)], axis=1)
        recon_data = tf.stack([pad_data[recon_shift[i]:recon_shift[i]+seq_len, i] for i in range(d)], axis=1)
        input_sv_mask = tf.stack([pad_sv_mask[input_shift[i]:input_shift[i]+seq_len, i] for i in range(d)], axis=1)
        recon_sv_mask = tf.stack([pad_sv_mask[recon_shift[i]:recon_shift[i]+seq_len, i] for i in range(d)], axis=1)
        #tf.print(input_data)
        #tf.print(input_data)
        return BatchInput(
            data=input_data,
            recon_data=recon_data, 
            sv_mask=input_sv_mask,
            recon_sv_mask=recon_sv_mask, 
            ext_input=batch_data.ext_input,
            dataset_id_ixs=batch_data.dataset_id_ixs)
        
        
    def apply_temporal_shift_v1(self, batch_data, cfg_node):
        """
        v1: relies on data being padded prior to modeling. Then adjusts the
        seq_len based on the temporal shift value. This creates a mismatch
        between the model seq_len and the data seq length in file.

        Shifts channels randomly in the temporal dimension.  This is useful to
        help the LFADS system avoid overfitting to individual spikes or fast
        oscillations found in the data that are irrelevant to behavior. A
        pure 'tabula rasa' approach would avoid this, but LFADS is sensitive
        enough to pick up dynamics that you may not want.

        Parameters
        ----------
        batch_data : lfads_tf2.tuples.BatchInput
            BatchInput object containing data to be 
            shifted as well as external inputs and sv_masks 
        cfg_node : yacs node
            Sub-node of main config cfg.TRAIN.DATA.AUGMENT
        tshift : float
            the maximum number of bins to shift channels in either 
            direction
        shift_dist : string
            specifies the type of random distribution to sample shifts
            default is a truncated normal distribution
        """

        tshift = cfg_node.TEMPORAL_SHIFT
        shift_dist = cfg_node.TEMPORAL_SHIFT_DIST
        # data/mask used for input to encoders
        
        data = batch_data.data
        sv_mask = batch_data.sv_mask # if we shift, we should shift the sv mask

        # data/mask used for evaluating neg. log likelihood cost
        recon_data = batch_data.recon_data
        recon_sv_mask = batch_data.sv_mask # if we shift, we should shift the sv mask
    
        ext_input = batch_data.ext_input # need to truncate external inputs to match seq length
        if tshift == 0:
            return batch_data

        t, d = data.shape
    
        # get current seq length and determine output seq length based on amount of temporal shift
        input_seq_len = t
        output_seq_len = input_seq_len - 2*tshift

        # generate different random shifts for data used for input/reconstruction 
        shift = generate_random_shift(d, tshift, shift_dist)
        nll_shift = generate_random_shift(d, tshift, shift_dist) 
        
        shifted_data = tf.stack([data[shift[i]:shift[i]+output_seq_len, i] for i in range(d)], axis=1)
        shifted_recon_data = tf.stack([recon_data[recon_shift[i]:recon_shift[i]+output_seq_len, i] for i in range(d)], axis=1)
        shifted_sv_mask = tf.stack([sv_mask[shift[i]:shift[i]+output_seq_len, i] for i in range(d)], axis=1)
        shifted_recon_sv_mask = tf.stack([recon_sv_mask[recon_shift[i]:recon_shift[i]+output_seq_len, i] for i in range(d)], axis=1)
        # truncate external inputs on both edges by tshift padding
        if ext_input is not None:
            truncated_ext_input = ext_input[tshift:-tshift,:]
        else:
            truncated_ext_input = None
        
        return BatchInput(
            data=shifted_data,
            nll_data=shifted_nll_data, 
            sv_mask=shifted_sv_mask,
            nll_sv_mask=shifted_nll_sv_mask, 
            ext_input=truncated_ext_input,
            dataset_id_ixs=batch_data.dataset_id_ixs)

def log_transform_enc_input(batch_data):
    
    data = batch_data.data
    nan_mask = tf.math.is_nan(data)
    # convert nans to 0 
    data = tf.where(nan_mask, tf.zeros_like(data), data)
    # log transform input to encoder
    data = tf.math.log(data)

    # convert -inf 
    data = tf.where(nan_mask, tf.fill(tf.shape(data), np.nan), data)
    
    return BatchInput(
        data=data,
        recon_data=batch_data.recon_data,
        sv_mask=batch_data.sv_mask,
        recon_sv_mask=batch_data.recon_sv_mask,
        ext_input=batch_data.ext_input,
        dataset_id_ixs=batch_data.dataset_id_ixs)

    
def shuffle_spikes_in_time(batch_data, cfg_node): 
    """
    A wrapper function around numpy_jitter that allows the numpy function to 
    be applied to tensors. Handles the application of jitter to a portion of 
    the BatchInput tuple via tf.map in models.py.

    Parameters
    ----------
    batch_data : lfads_tf2.tuples.BatchInput
        BatchInput object containing spike count data to be 
        shuffled as well as external inputs and sv_mask 
    w : float
        the maximum number of bins to shuffle spikes in either 
        direction

    Returns
    -------
    lfads_tf2.tuples.BatchInput
        BatchInput object containing shuffled spike data and 
        the unchanged external inputs and sv_mask.
    """

    # if jitter width is 0, then return the data unchanged
    w = cfg_node.JITTER_WIDTH
    if w == 0:
        return batch_data
    # define the jitter function
    jitter = lambda data: numpy_jitter(data, w)
    # turn the numpy impl into a tf function and apply to data
    shuffled_spikes = tf.numpy_function(jitter, [batch_data.data], tf.float32)
    
    # return a new BatchInput tuple with shuffled spikes
    return BatchInput(
        data=shuffled_spikes,
        recon_data=shuffled_spikes, 
        sv_mask=batch_data.sv_mask,
        recon_sv_mask=batch_data.sv_mask, 
        ext_input=batch_data.ext_input,
        dataset_id_ixs=batch_data.dataset_id_ixs)


def numpy_jitter(data_bxtxd, w):
    """
    Shuffle the spikes in the temporal dimension.  This is useful to
    help the LFADS system avoid overfitting to individual spikes or fast
    oscillations found in the data that are irrelevant to behavior. A
    pure 'tabula rasa' approach would avoid this, but LFADS is sensitive
    enough to pick up dynamics that you may not want.

    NOTE: This function cannot be applied directly to a tf.Dataset 
    because it is implemented using `numpy`.

    Parameters
    ----------
    data_bxtxd : numpy.array
        Numpy array in the shape Time x Neurons containing the spiking 
        data to apply jitter to.
    w : int
        the maximum number of bins to shuffle spikes in either 
        direction

    Returns
    -------
    S_bxtxd : numpy.array
        Numpy array in the same shape as data_bxtxd but containing 
        the jittered spikes.
    """

    # get the shape of the data, which should be time by neurons
    T, N = data_bxtxd.shape

    # if passed width is 0, then return the data unchanged
    if w == 0:
        return data_bxtxd

    # find nans
    nan_mask = np.isnan(data_bxtxd)
    # replace nans with 0
    data_bxtxd[nan_mask] = 0
    
    # determine the maximum spike count
    max_counts = int(np.max(data_bxtxd))
    # initialize an array to hold the shuffled spikes
    S_bxtxd = np.zeros([T,N])

    # Intuitively, shuffle spike occurances, 0 or 1, but since we have counts,
    # Do it over and over again up to the max count.
    for mc in range(1,max_counts+1):
        # pull out indices where there's a spike greater than mc
        idxs = np.nonzero(data_bxtxd >= mc)

        # initialize array where ones are found at indices
        # NOTE: the following 2 lines don't seem to be necessary
        data_ones = np.zeros_like(data_bxtxd)
        data_ones[data_bxtxd >= mc] = 1

        # the number of indices found
        nfound = len(idxs[0])
        # generate random shuffling increments between [-w, w]
        shuffles_incrs_in_time = np.random.randint(-w, w+1, size=nfound)

        # copy the indices
        shuffle_tidxs = idxs[0].copy()
        # add the shuffle to the indices
        shuffle_tidxs += shuffles_incrs_in_time

        # Reflect on the boundaries to not lose mass.
        # if the indices are negative, make them positive
        shuffle_tidxs[shuffle_tidxs < 0] = -shuffle_tidxs[shuffle_tidxs < 0]
        # if the indices are larger than the size of the array, move back within bounds
        shuffle_tidxs[shuffle_tidxs > T-1] = \
                                             (T-1)-(shuffle_tidxs[shuffle_tidxs > T-1] -(T-1))

        # for every new time index and the original neuron index
        for iii in zip(shuffle_tidxs, idxs[1]):
            # add a spike
            S_bxtxd[iii] += 1

    # return spikes as float             
    s_bxtxd = S_bxtxd.astype(np.float32)
    s_bxtxd[nan_mask] = np.nan
    return s_bxtxd

def generate_random_shift(d, tshift, shift_dist):
    """ returns 1-d vector of random shift indices with same dimension as input data"""
    #tf.random.set_seed(1994)
    if shift_dist=='normal':
        shifts = tf.random.truncated_normal([d], stddev=tshift/2)
        offset = tf.constant(tshift, dtype=tf.float32)
        shifts = tf.round(shifts) + offset
        shifts = tf.cast(shifts, tf.int32)
        #shifts = tf.cast(tf.round(tf.random.truncated_normal([d], stddev=tshift/2)) \
        #                + tf.constant(tshift, dtype=tf.float32), tf.int32)                    
    elif shift_dist=='uniform':
        shifts = tf.random_uniform([d], minval=0, maxval=2*tshift, dtype=tf.int32)
    return shifts



def shuffle_spikes_in_time_fast(batch_data, w):
    """DO NOT USE AS-IS
    
    A faster, but currently broken form of the spike jitter function.
    We haven't been able to implement the for-loop at the beginning in
    a form that can be traced with tf.function. A potential alternative
    is to vectorize this operation through something like the below 
    snippet (along with a few other downstream modifications), which 
    requires tf.repeat (only available in TF 2.1). 

    nz_indices = tf.where(data)
    counts = tf.gather_nd(data, nz_indices)
    all_indices = tf.repeat(nz_indices, counts, axis=0)
    
    We leave the implementation available in case we decide to fix it
    in the future.
    """

    if w == 0:
        return batch_data

    max_spike_ct = tf.reduce_max(data)
    all_indices = []
    for count in tf.range(1, max_spike_ct+1):
        indices = tf.where(data >= count)
        count_ixs = tf.cast(
            tf.fill([tf.shape(indices)[0], 1], count), tf.int64)
        all_indices.append(tf.concat([indices, count_ixs], axis=-1))
    all_indices = tf.concat(all_indices, axis=0)
    shifts = tf.random.uniform(
        [tf.shape(all_indices)[0]], 
        minval=-w, 
        maxval=w+1, # maxval is exclusive 
        dtype=tf.int64)
    # split up the indices to shift only the time index
    all_indices = tf.unstack(all_indices, axis=1)
    time_indices = all_indices[1]
    time_indices += shifts
    # reflect on the boundaries so we don't lose spikes
    B, T, N = tf.shape(data)
    # take care of negatives
    time_indices = tf.abs(time_indices) 
    # take care of positives
    oob_ixs = tf.where(time_indices > T-1)
    oob_values = tf.gather_nd(time_indices, oob_ixs)
    oob_values = 2*(T-1) - oob_values
    time_indices = tf.tensor_scatter_nd_update(
        time_indices, oob_ixs, oob_values)
    all_indices[1] = time_indices
    all_indices = tf.stack(all_indices, axis=-1)
    # create the summing tensor
    sum_tensor = tf.scatter_nd(
        all_indices, 
        tf.ones(tf.shape(all_indices)[0]), 
        [B, T, max_spike_ct])
    jittered_data = tf.reduce_sum(sum_tensor, axis=-1)

    return BatchInput(
        data=jittered_data, 
        sv_mask=batch_data.sv_mask, 
        ext_input=batch_data.ext_input)
