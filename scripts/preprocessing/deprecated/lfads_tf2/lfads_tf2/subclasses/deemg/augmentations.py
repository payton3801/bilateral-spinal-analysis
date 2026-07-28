"""Functions for augmenting EMG data

This module contains functions that modify EMG  data immediately
prior to passing it into the model. They are best used in combination
with the `tf.Dataset.map` function.
"""

import numpy as np 
import tensorflow as tf 
from lfads_tf2.subclasses.deemg.tuples import BatchInput

def apply_temporal_shift(batch_data, tshift=0, shift_dist='normal'):
    """
    Shifts channels randomly in the temporal dimension.  This is useful to
    help the LFADS system avoid overfitting to individual spikes or fast
    oscillations found in the data that are irrelevant to behavior. A
    pure 'tabula rasa' approach would avoid this, but LFADS is sensitive
    enough to pick up dynamics that you may not want.

    Parameters
    ----------
    batch_data : lfads_tf2.subclasses.deemg.tuples.TshiftBatchInput
        TshiftBatchInput object containing EMG data to be 
        shifted as well as external inputs and sv_masks 

    tshift : float
        the maximum number of bins to shift channels in either 
        direction
    shift_dist : string
        specifies the type of random distribution to sample shifts
        default is a truncated normal distribution
    """

    # data/mask used for input to encoders
    data = batch_data.data
    sv_mask = batch_data.sv_mask # if we shift, we should shift the sv mask

    # data/mask used for evaluating neg. log likelihood cost
    nll_data = batch_data.nll_data
    nll_sv_mask = batch_data.sv_mask # if we shift, we should shift the sv mask
    
    ext_input = batch_data.ext_input # need to truncate external inputs to match seq length
    
    if tshift == 0:
        return batch_data

    t, d = data.shape
    
    # get current seq length and determine output seq length based on amount of temporal shift
    input_seq_len = t
    output_seq_len = input_seq_len - 2*tshift

    # generate different random shifts for data used for input/reconstruction 
    #shifts = generate_random_shift(d*2, tshift, shift_dist)
    #shift, nll_shift = tf.split(shifts, num_or_size_splits=2)
    shift = generate_random_shift(d, tshift, shift_dist)
    nll_shift = generate_random_shift(d, tshift, shift_dist) 

    shifted_data = tf.stack([data[shift[i]:shift[i]+output_seq_len, i] for i in range(d)], axis=1)
    shifted_nll_data = tf.stack([nll_data[nll_shift[i]:nll_shift[i]+output_seq_len, i] for i in range(d)], axis=1)
    shifted_sv_mask = tf.stack([sv_mask[shift[i]:shift[i]+output_seq_len, i] for i in range(d)], axis=1)
    shifted_nll_sv_mask = tf.stack([nll_sv_mask[nll_shift[i]:nll_shift[i]+output_seq_len, i] for i in range(d)], axis=1)
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
        ext_input=truncated_ext_input)

def log_transform_enc_input(batch_data):
    
    data = batch_data.data
    # log transform input to encoder
    data = tf.math.log(data)

    return BatchInput(
        data=data,
        nll_data=batch_data.nll_data,
        sv_mask=batch_data.sv_mask,
        nll_sv_mask=batch_data.nll_sv_mask,
        ext_input=batch_data.ext_input)

#@profile    
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

def trim_temporal_shift_buffers(batch_data, tshift=0):
    """
    Shifts channels randomly in the temporal dimension.  This is useful to
    help the LFADS system avoid overfitting to individual spikes or fast
    oscillations found in the data that are irrelevant to behavior. A
    pure 'tabula rasa' approach would avoid this, but LFADS is sensitive
    enough to pick up dynamics that you may not want.

    Parameters
    ----------
    batch_data : lfads_tf2.subclasses.deemg.tuples.TshiftBatchInput
        TshiftBatchInput object containing EMG data to be 
        shifted as well as external inputs and sv_masks 

    tshift : float
        the maximum number of bins to shift channels in either 
        direction
    shift_dist : string
        specifies the type of random distribution to sample shifts
        default is a truncated normal distribution
    """

    # data/mask used for input to encoders
    data = batch_data.data
    sv_mask = batch_data.sv_mask # if we shift, we should shift the sv mask
    ext_input = batch_data.ext_input # need to truncate external inputs to match seq length
    
    if tshift == 0:
        return batch_data
    
    truncated_data = data[tshift:-tshift,:]
    truncated_sv_mask = sv_mask[tshift:-tshift,:]
    truncated_ext_input = ext_input[tshift:-tshift,:]
        
    return BatchInput(
        data=truncated_data,
        nll_data=truncated_data, 
        sv_mask=truncated_sv_mask,
        nll_sv_mask=truncated_sv_mask, 
        ext_input=truncated_ext_input)
