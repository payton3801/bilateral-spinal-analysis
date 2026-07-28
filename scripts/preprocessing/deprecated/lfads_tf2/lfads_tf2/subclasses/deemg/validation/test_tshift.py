from lfads_tf2.subclasses.deemg.augmentations import apply_temporal_shift, generate_random_shift
from lfads_tf2.subclasses.deemg.tuples import BatchInput
import numpy as np
import pdb
import tensorflow as tf
import matplotlib.pyplot as plt
import copy
b = 1; # batch
d = 30;  # dim
t = 40;  # time

tshift = 6; # tshift
shift_dist = 'normal'
t_w_buffer = t + 2*tshift

batch_data = np.zeros((b, t_w_buffer, d))

batch_data[:,20,:] = 1

sv_mask = copy.deepcopy(batch_data)
sv_mask[:,35,:] = 1

ext_input = np.zeros(batch_data.shape)

batch = BatchInput(
    data=np.squeeze(batch_data),
    nll_data=np.squeeze(batch_data),
    sv_mask=np.squeeze(sv_mask),
    nll_sv_mask=np.squeeze(sv_mask),
    ext_input=np.squeeze(ext_input))

#shift_batch_1 = apply_temporal_shift(batch, tshift, shift_dist)
#shift_batch_2 = apply_temporal_shift(batch, tshift, shift_dist)
#shifts = generate_random_shift(d, tshift, 'normal')

import tensorflow_probability as tfp
tfd = tfp.distributions

#dist = tfd.TruncatedNormal(loc=tf.zeros([d])+tshift, scale=tf.ones([d])*(tshift/2.0), low=0, high=tshift*2)
dist = None
#@profile
def test_func(batch_data, tshift, shift_dist, dist):
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
    
    if shift_dist=='normal':
        #shift = tf.cast(tf.round(dist.sample()), tf.int32)
        shift = tf.cast(tf.round(tf.random.truncated_normal([d], stddev=tshift/2)) \
                        + tf.constant(tshift, dtype=tf.float32), tf.int32)  
        nll_shift = tf.cast(tf.round(tf.random.truncated_normal([d], stddev=tshift/2)) \
                        + tf.constant(tshift, dtype=tf.float32), tf.int32) 
        #shift = tf.cast(tf.round(tf.random.truncated_normal([d])))
        #nll_shift = tf.cast(tf.round(dist.sample()), tf.int32)
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


for i in range(500):
    shift_batch_1 = test_func(batch, tshift, shift_dist, dist)
#shift_batch_2 = test_func(batch, tshift, shift_dist, dist)
#
# plt.figure()
# plt.subplot(121);
# plt.imshow(batch.data);
# plt.title('Input to Enc');
# plt.xlabel('Channels')
# plt.ylabel('Bins')
# plt.subplot(122);
# plt.imshow(batch.nll_data);
# plt.title('NLL Eval Data');
# plt.xlabel('Channels')

# plt.suptitle('BEFORE temporal shift [data] sample #0')

# #
# plt.figure()
# plt.subplot(121);
# plt.imshow(batch.sv_mask);
# plt.title('Input to Enc');
# plt.xlabel('Channels')
# plt.ylabel('Bins')
# plt.subplot(122);
# plt.imshow(batch.nll_sv_mask);
# plt.title('NLL Eval Data');
# plt.xlabel('Channels')

# plt.suptitle('BEFORE temporal shift [sv_mask] sample #0')

# plt.figure()
# # check that shift works
# plt.subplot(121);
# plt.imshow(shift_batch_1.data);
# plt.title('Input to Enc');
# plt.xlabel('Channels')
# plt.ylabel('Bins')
# plt.subplot(122);
# plt.imshow(shift_batch_1.nll_data);
# plt.title('NLL Eval Data');
# plt.xlabel('Channels')

# plt.suptitle('AFTER applying temporal shift #1 [data] sample #0')

# plt.figure()
# # check that shift works
# plt.subplot(121);
# plt.imshow(shift_batch_1.data);
# plt.title('Input to Enc');
# plt.xlabel('Channels')
# plt.ylabel('Bins')
# plt.subplot(122);
# plt.imshow(shift_batch_1.nll_data);
# plt.title('NLL Eval Data');
# plt.xlabel('Channels')

# plt.suptitle('AFTER applying temporal shift #1 [data] sample #1')

# plt.figure()
# # check that shift works
# plt.subplot(121);
# plt.imshow(shift_batch_1.sv_mask);
# plt.title('Input to Enc');
# plt.xlabel('Channels')
# plt.ylabel('Bins')
# plt.subplot(122);
# plt.imshow(shift_batch_1.nll_sv_mask);
# plt.title('NLL Eval Data');
# plt.xlabel('Channels')

# plt.suptitle('AFTER applying temporal shift #1 [sv_mask] sample #0')

# # check that shift is random each time function is called
# plt.figure()
# plt.subplot(121);
# plt.imshow(shift_batch_2.data);
# plt.title('Input to Enc');
# plt.xlabel('Channels')
# plt.ylabel('Bins')
# plt.subplot(122);
# plt.imshow(shift_batch_2.nll_data);
# plt.title('NLL Eval Data');
# plt.xlabel('Channels')

# plt.suptitle('AFTER applying temporal shift #2 [data] sample #0')


# pdb.set_trace()
