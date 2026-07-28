import os
from os import path
import numpy as np
import h5py

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import lfads_tf2
from lfads_tf2.models import LFADS
from lfads_tf2.initializers import variance_scaling
from lfads_tf2.regularizers import DynamicL2
from lfads_tf2.tuples import LoadableData, DecoderInput, LFADSOutput, \
    SamplingOutput, LFADSInput, BatchInput
from lfads_tf2.utils import load_data
from lfads_tf2.subclasses.multisession.defaults import get_cfg_defaults
from lfads_tf2.subclasses.multisession.initializers import PCRInitializer
from lfads_tf2.subclasses.multisession.utils import load_posterior_averages

class MultiSessionLFADS(LFADS):
    """Stitched LFADS model, which will model multiple sessions using 
       the same dynamics. Each session will have its own trainable read-in and 
       read-out matrix but share the same center LFADS model."""
    # Model building is in this function in the superclass for proper overloading
    def get_cfg_defaults(self):
        """Returns the MultiSessionLFADS configuration defaults.

        Returns
        -------
        yacs.config.CfgNode
            The default configuration for the model.
        """
        return get_cfg_defaults()

    def build_model_from_init_call(self, **kwargs):
        """Builds MultiSession LFADS model based on a configuration

        The superclass does most of the leg work, but this function adds
        a network for each dataset that reduces the dimensionality to that 
        specified by the config. This is done using a trained linear 
        transformation.
        Keyword args are identical to the superclass. Make sure to use 
        the defaults config from the subclass folder.
        """
        super(MultiSessionLFADS, self).build_model_from_init_call(**kwargs)

        # organize readins for init with PCR 
        # if self.cfg.MODEL.PCR_INIT: 
        filepath = path.join(self.cfg.TRAIN.DATA.DIR, 'pcr_alignment.h5')
        with h5py.File(filepath, 'r') as pcr_file:
            alignment_matrices = {k : pcr_file[k]['matrix'].value for k in self.ds_names}
            alignment_biases = {k : pcr_file[k]['bias'].value for k in self.ds_names}
        
        # add low-dim readin matrix for each dataset
        # stored as dictionary of ds name to lowD readin
        self.session_readins = {
            ds : Sequential([
                Dense(self.cfg.MODEL.ENC_INPUT_DIM,
                      kernel_initializer=PCRInitializer(
                          alignment_matrices[ds], matrix_type='weight'
                        ),
                      kernel_regularizer=DynamicL2(
                          scale= self.cfg.TRAIN.L2.READIN_SCALE
                        ),
                      bias_initializer=PCRInitializer(
                          alignment_biases[ds], matrix_type='bias'
                        ),
                      name='lowd_linear'),
            ])
            for ds in self.ds_names
        }
        
        # add readout matrix for each dataset 
        # stored as dictionary of ds name to readout layer
        # NOTE: overwrites the rate_linear variable in the superclass
        self.rate_linear = {
            ds : Dense(self.cfg.MODEL.DATA_DIM,
            kernel_initializer=PCRInitializer(
                alignment_matrices[ds], matrix_type='weight'
            ),
            bias_initializer=PCRInitializer(
                alignment_biases[ds], matrix_type='bias'
            ),
            name='rate_linear'
            )
            for ds in self.ds_names
        }

        # freeze readin matrices 
        if self.cfg.TRAIN.FIX_READIN: 
            self.freeze_pcr_matrices()

    def call(self, lfads_input, use_logrates=tf.constant(False)):
        """Passes each dataset through its respective low-dim readin, 
        followed by LFADS encoders.

        Overloads the function from the base class, then calls it after
        updating the value of the encoder input. Args and kwargs are 
        identical to the superclass.
        """
        # Unpack the input and compute the encoder input
        data, ext_input, dataset_name = lfads_input
        # array filled with NaNs to store lowD output
        enc_input = tf.fill(
            (tf.shape(data)[0], self.cfg.MODEL.SEQ_LEN, self.cfg.MODEL.ENC_INPUT_DIM), np.nan
        )
        # Use dataset names to determine lowD readin for each sample
        for k in self.session_readins.keys(): 
            # find where the samples are the same as the first lowD readin key
            dataset_inds = tf.where(dataset_name == k)
            # project those datasets through the appropriate readin 
            lowd_data = self.session_readins[k](
                tf.gather_nd(data, dataset_inds)
            )
            # store lowd input in appropriate indices 
            enc_input = tf.tensor_scatter_nd_update(enc_input, dataset_inds, lowd_data)
        # Create low-D input for LFADS
        lfads_input_lowd = LFADSInput(
            enc_input=enc_input, ext_input=ext_input, dataset_name=dataset_name)
        # Pass data through LFADS
        lfads_output = super(MultiSessionLFADS, self).call(
            lfads_input_lowd, use_logrates, dataset_name=dataset_name)
        return lfads_output

    def transform_factors_to_rates(self, factors, use_logrates=False, dataset_name=None):
        
        # array filled with NaNs to store rates
        logrates = tf.fill(
            (tf.shape(factors)[0], self.cfg.MODEL.SEQ_LEN, self.cfg.MODEL.DATA_DIM), np.nan
        )
        # Use dataset names to determine readout for each sample
        for k in self.rate_linear.keys(): 
            # find where the samples are the same as the first lowD readin key
            dataset_inds = tf.where(dataset_name == k)
            # project those datasets through the appropriate readout 
            rate_data = self.rate_linear[k](
                tf.gather_nd(factors, dataset_inds)
            )
            # store lowd input in appropriate indices 
            logrates = tf.tensor_scatter_nd_update(logrates, dataset_inds, rate_data)
       
        rates = tf.exp(logrates)

        return logrates if use_logrates else rates

    def weighted_l2_loss(self):
        """ Calculate the L2 loss for the MultiSessionLFADS model.

        This function computes the weighted L2 across all of the 
        recurrent kernels in LFADS, plus the readin kernels.
        """
        l2 = super(MultiSessionLFADS, self).weighted_l2_loss()
        # Add the L2 cost of the readin, averaged over elements
        for k in self.session_readins.keys():
            kernel = self.session_readins[k].get_layer('lowd_linear').kernel
            readin_size = tf.size(kernel, out_type=tf.float32)
            l2 += tf.reduce_sum(self.session_readins[k].losses) / readin_size
        return l2

    def posterior_sample_and_average_call(self, lfads_input, n_samples):
        """Passes data through low-dim readins before performing posterior
        sampling.

        Overloads the function from the base class, then calls it after
        updating the value of the encoder input. Args and kwargs are 
        identical to the superclass.
        """
        # Unpack the input and compute the encoder input
        data, ext_input, dataset_name = lfads_input
        # Use dataset names to determine lowD readin for each sample
        enc_input = np.zeros(
            (tf.shape(data)[0], self.cfg.MODEL.SEQ_LEN, self.cfg.MODEL.ENC_INPUT_DIM))
        for k in self.session_readins.keys(): 
            # find where the samples are the same as the first lowD readin key
            dataset_inds = tf.squeeze(tf.where(dataset_name == k))
            # project those datasets through the appropriate readin 
            lowd_data = self.session_readins[k](
                tf.gather(data, dataset_inds)
            )
            enc_input[dataset_inds, :, :] = lowd_data
        # Create low-D input for LFADS
        lfads_input_lowd = LFADSInput(
            enc_input=enc_input, ext_input=ext_input, dataset_name=dataset_name)
        # Perform posterior sampling based on low-dim input
        output, non_averaged_outputs = super(MultiSessionLFADS, self) \
            .posterior_sample_and_average_call(lfads_input_lowd, n_samples)

        return output, non_averaged_outputs

    def load_posterior_averages(self, model_dir, merge_tv=False): 
        return load_posterior_averages(model_dir, merge_tv=merge_tv)

    def get_sampling_output(self, outputs, dataset_names, output_file, prefix, save=True):
        # get the sampling output 
        samp_out = super(MultiSessionLFADS, self).get_sampling_output(
            outputs, dataset_names, output_file, prefix, save=False
        )

        dataset_names = np.concatenate(dataset_names).astype('str')

        new_out = {k : [] for k in np.unique(dataset_names)} # each list will hold divided output

        # use the dataset names to reorganize output by dataset
        for field in samp_out._fields: 
            data = getattr(samp_out, field)
            if tf.is_tensor(data):
                # if a tensor
                data = data.numpy()
            
            for k in np.unique(dataset_names):
                where_dataset = np.where(dataset_names == k)[0]
                if len(data.shape) == 3:
                    new_out[k].append(data[where_dataset, :, :])
                elif len(data.shape) == 2:
                    new_out[k].append(data[where_dataset, :])
                elif len(data.shape) == 1:
                    # ic priors 
                    new_out[k].append(data)

        new_out = [SamplingOutput._make(new_out[k]) for k in new_out.keys()]
        # writes the output to the a file in the model directory
        available_datasets = self.train_inds.keys()
        with h5py.File(output_file, 'a') as hf:
            for ii, k in enumerate(available_datasets):
                group = hf.create_group(prefix+k)
                curr_out = new_out[ii]
                output_fields = list(samp_out._fields)
                for field in output_fields:
                    group.create_dataset(
                        prefix+field,
                        data=getattr(curr_out, field).astype('float'))
                group.create_dataset('train_inds', data=self.train_inds[k])
                group.create_dataset('valid_inds', data=self.valid_inds[k])

        return new_out
        
    def freeze_pcr_matrices(self): 
        data_shape, _, ext_input_shape, name_shape = self.get_input_shapes(10)

        remainder = name_shape[0] - name_shape[0]//len(self.ds_names) * len(self.ds_names)
        dataset_names = name_shape[0]//len(self.ds_names) * self.ds_names + self.ds_names[:remainder]

        noise = LFADSInput(
            enc_input=np.ones(shape=data_shape, dtype=np.float32),
            ext_input=np.ones(shape=ext_input_shape, dtype=np.float32),
            dataset_name=np.array(dataset_names)
        )

        self.call(noise) # initialize weights so they can be overwritten 
        
        for k in self.ds_names: 
            try:
                for var in self.session_readins[k].trainable_variables: 
                    var._trainable = False
            except: 
                pass
        # for var in self.session_readins.trainable_variables: 
            
