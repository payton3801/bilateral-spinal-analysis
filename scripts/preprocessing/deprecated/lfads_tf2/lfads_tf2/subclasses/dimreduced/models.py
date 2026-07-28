import os
import pickle
import numpy as np
from os import path

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow_addons.layers import GroupNormalization

from lfads_tf2.models import LFADS
from lfads_tf2.initializers import variance_scaling
from lfads_tf2.regularizers import DynamicL2
from lfads_tf2.tuples import LFADSInput

from lfads_tf2.subclasses.dimreduced.defaults import get_cfg_defaults
from lfads_tf2.subclasses.dimreduced.factor_analysis import (
    get_factor_analysis_loading, get_stabilization_matrices
)

class DimReducedLFADS(LFADS):
    """Modified LFADS model with low-D read-in network."""
    def get_cfg_defaults(self):
        """Returns the DimReducedLFADS configuration defaults.

        Returns
        -------
        yacs.config.CfgNode
            The default configuration for the model.
        """
        return get_cfg_defaults()

    def build_model_from_init_call(self, **kwargs):
        """Builds DimReduced LFADS model based on a configuration

        The superclass does most of the leg work, but this function adds
        a network that can optionally be initialized using factor analysis.
        Keyword args are identical to the superclass. Make sure to use 
        the defaults config from the subclass folder.
        """
        super(DimReducedLFADS, self).build_model_from_init_call(**kwargs)
        # Add linear mapping followed by layer normalization
        self.lowd_readin = Sequential([
            Dense(
                self.cfg.MODEL.ENC_INPUT_DIM,
                kernel_initializer=variance_scaling,
                kernel_regularizer=DynamicL2(
                scale=self.cfg.TRAIN.L2.READIN_SCALE),
                name='lowd_linear'),
            GroupNormalization(groups=1, axis=-1, epsilon=1e-12)
        ])
        # Optionally initialize the mapping using factor analysis loading
        if self.cfg.MODEL.SMART_INIT and not self.from_existing: 
            self.init_lowd_readin_from_fa()
        elif self.cfg.MODEL.SMART_INIT and self.from_existing: 
            loading_matrix_path = path.join(
                self.cfg.TRAIN.MODEL_DIR, 'loading_matrix.pkl')
            with open(loading_matrix_path, 'rb') as f:
                self.loading = pickle.load(f)

    def call(self, lfads_input, use_logrates=tf.constant(False)):
        """Passes data through low-dim readin, followed by LFADS encoders.

        Overloads the function from the base class, then calls it after
        updating the value of the encoder input. Args and kwargs are 
        identical to the superclass.
        """
        # Unpack the input and compute the encoder input
        data, ext_input, dataset_name = lfads_input
        enc_input = self.lowd_readin(data)
        # Create low-D input for LFADS
        lfads_input_lowd = LFADSInput(
            enc_input=enc_input, ext_input=ext_input, dataset_name=dataset_name)
        # Pass data through LFADS
        lfads_output = super(DimReducedLFADS, self).call(
            lfads_input_lowd, use_logrates)

        return lfads_output

    def posterior_sample_and_average_call(self, lfads_input, n_samples):
        """Passes data through low-dim readin before performing posterior
        sampling.

        Overloads the function from the base class, then calls it after
        updating the value of the encoder input. Args and kwargs are 
        identical to the superclass.
        """
        # Unpack the input and compute the encoder input
        data, ext_input, dataset_name = lfads_input
        enc_input = self.lowd_readin(data)
        # Create low-D input for LFADS
        lfads_input_lowd = LFADSInput(
            enc_input=enc_input, ext_input=ext_input, dataset_name=dataset_name)
        # Perform posterior sampling based on low-dim input
        output, non_averaged_outputs = super(DimReducedLFADS, self) \
            .posterior_sample_and_average_call(lfads_input_lowd, n_samples)

        return output, non_averaged_outputs

    # def batch_to_LFADSInput(self, batch):
    #     """Converts a BatchInput named tuple to an LFADSInput named tuple.

    #     Parameters
    #     ----------
    #     batch : lfads_tf2.tuples.BatchInput
    #         A namedtuple contining tf.Tensors for spiking data, 
    #         external inputs, and a sample validation mask.
    #     Returns
    #     -------
    #     lfads_tf2.tuples.LFADSInput
    #         A namedtuple contining tf.Tensors for spiking data, external input, and encoder_input
    #     """
    #     # Unpack the input and compute the encoder input
    #     data, ext_input, _ = batch
    #     enc_input = self.lowd_readin(data)
    #     # Create the low-D input for LFADS
    #     lfads_input_lowd = LFADSInput(
    #         enc_input=enc_input, ext_input=ext_input)
    #     return lfads_input_lowd

    def weighted_l2_loss(self):
        """ Calculate the L2 loss for the DimReducedLFADS model.

        This function computes the weighted L2 across all of the 
        recurrent kernels in LFADS, plus the readin kernel.
        """
        l2 = super(DimReducedLFADS, self).weighted_l2_loss()
        # Add the L2 cost of the readin, averaged over elements
        kernel = self.lowd_readin.get_layer('lowd_linear').kernel
        readin_size = tf.size(kernel, out_type=tf.float32)
        l2 += tf.reduce_sum(self.lowd_readin.losses) / readin_size
        return l2

    def load_matrix_to_lowd_readin(self, matrix, bias=None, freeze=False): 
        '''
        Assigns an input matrix to the weights of the lowD readin
        
        matrix: numpy.array
            array containing values to assign as lowD readin 
            weights
        freeze: bool
            whether to freeze these weights for training, 
            by default False
        '''
        data_shape, _, ext_input_shape, name_shape = self.get_input_shapes(10)
        noise = LFADSInput(
            enc_input=np.ones(shape=data_shape, dtype=np.float32),
            ext_input=np.ones(shape=ext_input_shape, dtype=np.float32),
            dataset_name=np.full(shape=name_shape, fill_value='')
        )
        self.call(noise) # initialize weights so they can be overwritten 

        # assign the sequential kernel to hold the matrix as weights 
        self.lowd_readin.weights[0].assign(matrix)

        # if a bias term is provided, assign this here 
        if bias is not None: 
            self.lowd_readin.weights[1].assign(bias)
        
        # freeze all assigned values values 
        if freeze: 
            self.lowd_readin.trainable_variables[0]._trainable = False
            
            if bias is not None: 
                self.lowd_readin.trainable_variables[1]._trainable = False

    def init_lowd_readin_from_fa(self): 
        '''Creates and loads a matrix to the lowD readin'''
        # get FA loading matrix
        loading, psi, d = \
            get_factor_analysis_loading(self.train_tuple.data, 
                                        self.cfg.MODEL.ENC_INPUT_DIM, 
                                        n_restarts=5)
        self.loading = loading
        
        # save loading matrix to file for alignment 
        loading_matrix_path = path.join(self.cfg.TRAIN.MODEL_DIR, 'loading_matrix.pkl')
        f = open(loading_matrix_path, 'wb')
        pickle.dump(loading, f)
        f.close()

        # get beta and o terms to make factors a linear transformation
        beta, o = get_stabilization_matrices(loading, psi, d)
        # assign beta matrix to lowD readin, o to bias, and freeze 
        # need to transpose beta due to dimensionality 
        self.load_matrix_to_lowd_readin(beta.T, bias=o, freeze=self.cfg.TRAIN.FIX_READIN)
