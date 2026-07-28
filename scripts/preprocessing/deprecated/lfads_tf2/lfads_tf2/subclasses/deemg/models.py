import os
from os import path
import numpy as np
import h5py

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import Progbar
import tensorflow_probability as tfp
tfd = tfp.distributions

import lfads_tf2
from lfads_tf2.models import LFADS
from lfads_tf2.initializers import variance_scaling
from lfads_tf2.regularizers import DynamicL2
from lfads_tf2.tuples import LoadableData, DecoderInput, LFADSInput
from lfads_tf2.utils import load_data
from lfads_tf2.subclasses.deemg.tuples import BatchInput, deEMGOutput, SamplingOutput
from lfads_tf2.subclasses.deemg.defaults import get_cfg_defaults
from lfads_tf2.subclasses.deemg.augmentations import apply_temporal_shift, trim_temporal_shift_buffers, log_transform_enc_input

class deEMG(LFADS):
    """deEMG model, which will model EMG data using gamma noise emissions
    but use the same core LFADS model."""
    # Model building is in this function in the superclass for proper overloading
    def get_cfg_defaults(self):
        """Returns the deEMG configuration defaults.

        Returns
        -------
        yacs.config.CfgNode
            The default configuration for the model.
        """
        return get_cfg_defaults()

    def build_model_from_init_call(self, **kwargs):
        """Builds deEMG  model based on a configuration

        The superclass does most of the leg work, but this function adds
        a network for each dataset that reduces the dimensionality to that 
        specified by the config. This is done using a trained linear 
        transformation.
        Keyword args are identical to the superclass. Make sure to use 
        the defaults config from the subclass folder.
        """
        super(deEMG, self).build_model_from_init_call(**kwargs)

        # construct separate linear transformations for each of the Gamma dist. params
        self.rate_linear_alpha = Dense(self.cfg.MODEL.DATA_DIM,
                                       kernel_initializer=variance_scaling,
                                       name='rate_linear_alpha')
        self.rate_linear_beta = Dense(self.cfg.MODEL.DATA_DIM,
                                       kernel_initializer=variance_scaling,
                                       name='rate_linear_beta')

        
    def call(self, lfads_input):
        """Passes each dataset through its respective low-dim readin, 
        followed by LFADS encoders.

        Overloads the function from the base class, then calls it after
        updating the value of the encoder input. Args and kwargs are 
        identical to the superclass.
        """
        # Unpack the input and compute the encoder input
        enc_input, ext_input = lfads_input
        
        # encode spikes into generator IC distributions and controller inputs
        ic_mean, ic_stddev, ci = self.encoder(enc_input, training=self.training)
        # pass one sample from each IC posterior through the decoder network
        ic_post = tfd.MultivariateNormalDiag(ic_mean, ic_stddev)
        dec_input = DecoderInput(
            ic_samp=ic_post.sample(),
            ci=ci,
            ext_input=ext_input)
        dec_output = self.decoder(
            dec_input,
            training=self.training,
            # use_logrates=use_logrates
        )
        co_mean, co_stddev, factors, gen_states, \
            gen_init, gen_inputs, con_states = dec_output

        alpha, beta = self.transform_factors_to_dist_params(factors)
        
        rates = tfd.Gamma(alpha,beta).mean()

        return deEMGOutput(
            rates=rates,
            alpha=alpha,
            beta=beta,
            ic_means=ic_mean,
            ic_stddevs=ic_stddev,
            co_means=co_mean,
            co_stddevs=co_stddev,
            factors=factors,
            gen_states=gen_states,
            gen_init=gen_init,
            gen_inputs=gen_inputs,
            con_states=con_states)

    def transform_factors_to_dist_params(self, factors):
        log_alpha = self.rate_linear_alpha(factors)
        log_beta = self.rate_linear_beta(factors)
        alpha = tf.exp(log_alpha)
        beta = tf.exp(log_beta)

        return alpha, beta
                
    def load_datasets_from_arrays(self, loadable_data):
        """Creates TF datasets and attaches them to the LFADS object.

        This function builds dataset objects from input arrays.
        The datasets are used for shuffling, batching, data 
        augmentation, and more. These datasets are used by 
        both posterior sampling and training functions.

        Parameters
        ----------
        loadable_data : lfads_tf2.tuples.LoadableData
            A namedtuple containing the input data and external
            inputs for the training and validation sets. See 
            definition of LoadableData for more detail.

        See Also
        --------
        lfads_tf2.models.LFADS.load_datasets_from_file : 
            A wrapper around this function that loads from a file.

        """

        train_data, valid_data, train_ext, valid_ext, \
            train_inds, valid_inds = loadable_data

        if train_ext is None or valid_ext is None:
            # use empty tensors if there are no inputs
            train_ext = tf.zeros(train_data.shape[:-1] + (0,))
            valid_ext = tf.zeros(valid_data.shape[:-1] + (0,))
        # create the sample validation masks
        sv_seed = self.cfg.MODEL.SV_SEED
        train_sv_mask = self.sv_input_dist.sample(
            sample_shape=tf.shape(train_data), seed=sv_seed)
        valid_sv_mask = self.sv_input_dist.sample(
            sample_shape=tf.shape(valid_data), seed=sv_seed)
        # package up the data into tuples and use to build datasets
        self.train_tuple = BatchInput(
            data=train_data,
            nll_data=train_data,
            sv_mask=train_sv_mask,
            nll_sv_mask=train_sv_mask,
            ext_input=train_ext)
        self.valid_tuple = BatchInput(
            data=valid_data,
            nll_data=valid_data,
            sv_mask=valid_sv_mask,
            nll_sv_mask=valid_sv_mask,
            ext_input=valid_ext)
        # create the datasets to batch the data, masks, and input
        self._train_ds = tf.data.Dataset.from_tensor_slices(self.train_tuple)
        self._valid_ds = tf.data.Dataset.from_tensor_slices(self.valid_tuple)
                  
        if self.cfg.TRAIN.BATCH_SIZE > len(self.train_tuple.data):
            drop_remainder = False
        else:
            drop_remainder = True
            
        self.train_dataset, self.valid_dataset = self.build_dataset_pipelines(drop_remainder)
        
        # save the indices
        self.train_inds, self.valid_inds = train_inds, valid_inds

    def batch_call(self, batch):
        """Performs the forward pass on a batch of input data.

        This is a wrapper around the forward pass of LFADS, meant to be 
        more compatible with the coordinated dropout and sample 
        validation wrappers.

        Parameters
        ----------
        batch : lfads_tf2.tuples.BatchInput
            A namedtuple contining tf.Tensors for spiking data, 
            external inputs, and a sample validation mask.

        Returns
        -------
        tf.Tensor
            A BxTxN tensor of log-rates, where B is the batch size, 
            T is the number of time steps, and N is the number of neurons.
        tuple of tf.Tensor
            Four tensors corresponding to the posteriors - `ic_mean`, 
            `ic_stddev`, `co_mean`, `co_stddev`.
        tf.Tensor
            A BxTxN boolean tensor that indicates whether matrix elements 
            should be used to calculate gradients.

        """
        lfads_input = self.batch_to_LFADSInput(batch)
        if self.cfg.TRAIN.EAGER_MODE:
            output = self.call(lfads_input)
        else:
            output = self.graph_call(lfads_input)

        posterior_params = (
            output.ic_means,
            output.ic_stddevs,
            output.co_means,
            output.co_stddevs)

        return output.alpha, output.beta, posterior_params
            
    def add_sample_validation(self, model_call):
        """Applies sample validation to a model-calling function.
        
        This decorator applies sample validation to a forward pass 
        through the model. It sets a certain proportion of the input 
        data elements to zero (heldout) and scales up the remaining 
        elements (heldin). It then computes NLL on the heldin 
        samples and the heldout samples separately. `nll_heldin` is 
        used for optimization and `nll_heldout` is used as a metric 
        to detect overfitting to spikes.

        Parameters
        ----------
        model_call : callable
            A callable function with inputs and outputs identical to 
            `LFADS.batch_call`.
        
        Returns
        -------
        callable
            A wrapper around `model_call` that computes heldin and 
            heldout NLL and returns the posterior parameters.
        """

        def sv_step(batch):
            # unpack the batch
            data, nll_data, sv_mask, nll_sv_mask, *_ = batch
            heldin_mask = sv_mask
            heldout_mask = tf.logical_not(nll_sv_mask)
            # set the heldout data to zero and scale up heldin data
            wt_mask = tf.cast(heldin_mask, tf.float32) / self.sv_keep
            heldin_data = data * wt_mask
            # perform the forward pass on the heldin data
            new_batch = BatchInput(
                data=heldin_data,
                nll_data=batch.nll_data,
                sv_mask=batch.sv_mask,
                nll_sv_mask=batch.nll_sv_mask,
                ext_input=batch.ext_input)
            alpha, beta, posterior_params = model_call(new_batch)
            # compute the nll of the observed samples
            nll_heldin = self.neg_log_likelihood(nll_data, alpha, beta, wt_mask)
            if self.sv_keep < 1:
                # exclude the observed samples from the nll_heldout calculation
                wt_mask = tf.cast(heldout_mask, tf.float32) / (1-self.sv_keep)
                nll_heldout = self.neg_log_likelihood(nll_data, alpha, beta, wt_mask)
            else:
                nll_heldout = np.nan

            return nll_heldin, nll_heldout, posterior_params

        return sv_step


    def add_coordinated_dropout(self, model_call):
        """Applies coordinated dropout to a model-calling function.
        
        A decorator that applies coordinated dropout to a forward pass 
        through the model. It sets a certain proportion of the input 
        data elements to zero and scales up the remaining elements. When 
        the model is being trained, it can only backpropagate gradients 
        for matrix elements it didn't see at the input. The function 
        outputs a gradient mask that is incorporated by the sample 
        validation wrapper.

        Parameters
        ----------
        model_call : callable
            A callable function with inputs and outputs identical to 
            `LFADS.batch_call`.
        
        Returns
        -------
        callable
            A wrapper around `model_call` that blocks matrix elements 
            before the call and passes a mask to block gradients of the 
            observed matrix elements.

        """

        def block_gradients(input_data, keep_mask):
            keep_mask = tf.cast(keep_mask, tf.float32)
            block_mask = 1 - keep_mask
            return tf.stop_gradient(input_data * block_mask) + input_data * keep_mask

        def cd_step(batch):
            input_data = batch.data
            # samples a new coordinated dropout mask at every training step
            cd_mask = self.cd_input_dist.sample(sample_shape=tf.shape(input_data))
            pass_mask = self.cd_pass_dist.sample(sample_shape=tf.shape(input_data))
            grad_mask = tf.logical_or(tf.logical_not(cd_mask), pass_mask)
            # mask and scale up the post-CD input so it has the same sum as the original data
            cd_masked_data = input_data * tf.cast(cd_mask, tf.float32)
            cd_masked_data /= self.cd_keep
            # perform a forward pass on the cd masked data
            new_batch = BatchInput(
                data=cd_masked_data,
                nll_data=cd_masked_data, # doesn't matter just need to pass something
                sv_mask=batch.sv_mask,
                nll_sv_mask=batch.sv_mask, # doesn't matter just need to pass something
                ext_input=batch.ext_input)
            alpha, beta, posterior_params = model_call(new_batch)
            # block the gradients with respect to the masked outputs
            alpha = block_gradients(alpha, grad_mask)
            beta = block_gradients(beta, grad_mask)
            return alpha, beta, posterior_params

        return cd_step

    def _build_wrapped_call(self):
        """Assembles the forward pass using SV and CD wrappers.

        Conveniently wraps the forward pass of LFADS with coordinated
        dropout and sample validation to allow automatic application 
        of these paradigms.
        """
        train_call = self.batch_call
        if self.cd_keep < 1:
            train_call = self.add_coordinated_dropout(train_call)
        train_call = self.add_sample_validation(train_call)
        val_call = self.add_sample_validation(self.batch_call)

        if self.cfg.TRAIN.EAGER_MODE:
            self.train_call = train_call
            self.val_call = val_call
        else:
            data_shape, _, ext_input_shape  = self.get_input_shapes()
            # single step of training or validation
            input_signature=[
                BatchInput(
                    tf.TensorSpec(shape=data_shape), # data used for reconstruction
                    tf.TensorSpec(shape=data_shape), # data used for reconstruction
                    tf.TensorSpec(shape=data_shape, dtype=tf.bool), # mask for sample validation
                    tf.TensorSpec(shape=data_shape, dtype=tf.bool), # mask for sample validation
                    tf.TensorSpec(shape=ext_input_shape))] # external inputs
            self.train_call = tf.function(func=train_call, input_signature=input_signature)
            self.val_call = tf.function(func=val_call, input_signature=input_signature)

            
    def neg_log_likelihood(self, data, alpha, beta, wt_mask=None):
        """Computes the Gamma log likelihood of the data, given 
        predicted rates. 

        This function computes the average negative log likelihood 
        of the EMG in this batch, given the alpha/beta that LFADS 
        predicts for the samples.

        Parameters
        ----------
        data : tf.Tensor
            A BxTxN tensor of spiking data.
        alpha : tf.Tensor
            A BxTxN tensor of alpha.
        beta : tf.Tensor
            A BxTxN tensor of beta.
        wt_mask : tf.Tensor
            A weighted mask to apply to the likelihoods.

        Returns
        -------
        tf.Tensor
            A scalar tensor representing the mean negative 
            log-likelihood of these EMG.        
        
        """
        if wt_mask is None:
            wt_mask = tf.ones_like(data)
        nll_all = -tfd.Gamma(alpha, beta).log_prob(data)
        nll_masked = nll_all * wt_mask
        if self.cfg.TRAIN.NLL_MEAN:
            # Average over all elements of the data tensor
            nll = tf.reduce_mean(nll_masked)
        else:
            # Sum over inner dimensions, average over batch dimension
            nll = tf.reduce_mean(tf.reduce_sum(nll_masked, axis=[1,2]))
        return nll

    def build_dataset_pipelines(self, drop_remainder=False):
        """Builds pipelines for train and valid datasets for train_epoch

        This function defines the data augmentation operation
        to perform during training.
        
        Returns
        -------
        train_dataset : tf.data.Dataset
        valid_dataset : tf.data.Dataset
        """
        
        # use autotune to parallelize computations automatically
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        # apply temporal shift
        tshift = self.cfg.TRAIN.DATA.AUGMENT.TEMPORAL_SHIFT
        shift_dist = self.cfg.TRAIN.DATA.AUGMENT.TEMPORAL_SHIFT_DIST
        augment = lambda batch : apply_temporal_shift(batch, tshift, shift_dist)
        transform_enc_input = lambda batch : log_transform_enc_input(batch)
        train_dataset = (
                    self._train_ds
                    # apply augmentation to the training data
                    .map(augment, num_parallel_calls=AUTOTUNE)
                    .map(transform_enc_input, num_parallel_calls=AUTOTUNE)
                    # shuffle samples with buffer size > # of samples
                    .shuffle(10000)
                    # divide into batches
                    .batch(self.cfg.TRAIN.BATCH_SIZE, drop_remainder=drop_remainder)
                    #.prefetch(2)
                )
        valid_dataset = (
                    self._valid_ds
                    # apply jitter to valid data
                    .map(augment, num_parallel_calls=AUTOTUNE)
                    .map(transform_enc_input, num_parallel_calls=AUTOTUNE)            
                    # divide into batches
                    .batch(self.cfg.TRAIN.BATCH_SIZE)
                    #.prefetch(2)
                )
        
        return train_dataset, valid_dataset

    def build_graph(self):
        # ===== AUTOGRAPH FUNCTIONS =====
        # compile the `_step` function into a graph for better speed
        data_shape, _, ext_input_shape = self.get_input_shapes()
        # single step of training or validation
        self._graph_step = tf.function(
            func=self._step,
            input_signature=[
                BatchInput(
                    tf.TensorSpec(shape=data_shape), # data used for reconstruction
                    tf.TensorSpec(shape=data_shape), # data used for reconstruction
                    tf.TensorSpec(shape=data_shape, dtype=tf.bool), # mask for sample validation
                    tf.TensorSpec(shape=data_shape, dtype=tf.bool), # mask for sample validation
                    tf.TensorSpec(shape=ext_input_shape))]) # external inputs
        
        # forward pass of LFADS
        self.graph_call = tf.function(
            func=self.call,
            input_signature=[
                LFADSInput(
                    tf.TensorSpec(shape=data_shape),
                    tf.TensorSpec(shape=ext_input_shape))])
                    
    def posterior_sample_and_average_call(self, lfads_input, n_samples):
        """ Performs the posterior estimation for the LFADS graph using 
        the input data.
        
        NOTE: Overloadable

        Parameters
        ----------
        lfads_input : lfads_tf2.tuples.LFADSInput
            A namedtuple of tensors containing the data, external inputs, 
            and encoder inputs.
        n_samples : int
            The number of samples to take from the posterior 
            distribution for each datapoint, by default 50.

        Returns
        -------
        list of np.ndarray
            Things that are averaged across samples (most things)
        list of np.ndarray
            IC means and stddevs
        """
        # Unpack the input data
        enc_input, ext_input = lfads_input
        #enc_input = tf.math.log(enc_input)
        
        # for each chop in the dataset, compute the initial conditions distribution
        ic_mean, ic_stddev, ci = self.encoder.graph_call(enc_input)
        ic_post = tfd.MultivariateNormalDiag(ic_mean, ic_stddev)

        # define merging and splitting utilities
        def merge_samp_and_batch(data, batch_dim):
            """ Combines the sample and batch dimensions """
            return tf.reshape(
                data, [n_samples * batch_dim] + tf.unstack(tf.shape(data)[2:]))

        def split_samp_and_batch(data, batch_dim):
            """ Splits up the sample and batch dimensions """
            return tf.reshape(
                data, [n_samples, batch_dim] + tf.unstack(tf.shape(data)[1:]))

        # sample from the posterior and merge sample and batch dimensions
        ic_post_samples = ic_post.sample(n_samples)
        ic_post_samples_merged = merge_samp_and_batch(
            ic_post_samples, len(enc_input))

        # tile and merge the controller inputs and the external inputs
        ci_tiled = tf.tile(tf.expand_dims(ci, axis=0), [n_samples, 1, 1, 1])
        ci_merged = merge_samp_and_batch(ci_tiled, len(enc_input))
        ext_tiled = tf.tile(tf.expand_dims(ext_input, axis=0), [n_samples, 1, 1, 1])
        ext_merged = merge_samp_and_batch(ext_tiled, len(enc_input))

        # pass all samples into the decoder
        dec_input = DecoderInput(
            ic_samp=ic_post_samples_merged,
            ci=ci_merged,
            ext_input=ext_merged)
        dec_output = self.decoder.graph_call(dec_input)

        co_mean, co_stddev, factors, gen_states, \
            gen_init, gen_inputs, con_states = dec_output

        alpha, beta = self.transform_factors_to_dist_params(factors)
        rates = tfd.Gamma(alpha,beta).mean()

        output_samples_merged = [rates,alpha,beta,co_mean,co_stddev, \
                                 factors,gen_states,gen_init,gen_inputs, \
                                 con_states] 
        #output_samples_merged = self.decoder.graph_call(dec_input)
        
        # average the outputs across samples
        output_samples = [split_samp_and_batch(t, len(enc_input)) \
            for t in output_samples_merged]
        output = [np.mean(t, axis=0) for t in output_samples]

        # aggregate for each batch
        non_averaged_outputs = [
            ic_mean.numpy(),
            tf.math.log(ic_stddev**2).numpy(),
        ]
        
        return output, non_averaged_outputs

    def sample_and_average(self,
                           loadable_data=None,
                           n_samples=50,
                           batch_size=64,
                           ps_filename='posterior_samples.h5',
                           save=True,
                           merge_tv=False):
        """Saves rate estimates to the 'model_dir'.
        
        Performs a forward pass of LFADS, but passes multiple 
        samples from the posteriors, which can be used to get a 
        more accurate estimate of the rates. Saves all output 
        to posterior_samples.h5 in the `model_dir`.

        Parameters
        ----------
        loadable_data : lfads_tf2.tuples.LoadableData, optional
            A namedtuple containing the input data and external
            inputs for the training and validation sets. See 
            definition of LoadableData for more detail. By default,
            None uses data that has already been loaded.
        n_samples : int, optional
            The number of samples to take from the posterior 
            distribution for each datapoint, by default 50.
        batch_size : int, optional
            The number of samples per batch, by default 128.
        ps_filename : str, optional
            The name of the posterior sample file, by default
            'posterior_samples.h5'. Ignored if `save` is False.
        save : bool, optional
            Whether or not to save the posterior sampling output
            to a file, if False will return a tuple of 
            SamplingOutput. By default, True.
        merge_tv : bool, optional
            Whether to merge training and validation output, 
            by default False. Ignored if `save` is True.

        Returns
        -------
        SamplingOutput
            If save is True, return nothing. If save is False, 
            and merge_tv is false, retun SamplingOutput objects 
            training and validation data. If save is False and 
            merge_tv is True, return a single SamplingOutput 
            object.

        """

        if loadable_data is not None:
            self.load_datasets_from_arrays(loadable_data)
        
        # get the filename for posterior sampling output
        output_file = path.join(
            self.cfg.TRAIN.MODEL_DIR, ps_filename)

        try:
            # remove any pre-existing posterior sampling file
            os.remove(output_file)
            self.lgr.info(
                f"Removing existing posterior sampling file at {output_file}")
        except OSError:
            pass

        if not self.is_trained:
            self.lgr.warn(
                "Performing posterior sampling on an untrained model.")

        # ========== POSTERIOR SAMPLING ==========
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        tshift = self.cfg.TRAIN.DATA.AUGMENT.TEMPORAL_SHIFT
        augment = lambda batch : trim_temporal_shift_buffers(batch, tshift)
        transform_enc_input = lambda batch : log_transform_enc_input(batch)
        # perform sampling on both training and validation data
        for prefix, dataset in zip(['train_', 'valid_'], [self._train_ds, self._valid_ds]):
            data_len = len(self.train_tuple.data) if prefix == 'train_' else len(self.valid_tuple.data)

            # initialize lists to store rates
            all_outputs = []
            self.lgr.info("Posterior sample and average on {} segments.".format(data_len))
            if not self.cfg.TRAIN.TUNE_MODE:
                pbar = Progbar(data_len, width=50, unit_name='dataset')
            for batch in dataset.map(augment, num_parallel_calls=AUTOTUNE).map(transform_enc_input, num_parallel_calls=AUTOTUNE).batch(batch_size):
                # convert the batch into LFADS input
                lfads_input = self.batch_to_LFADSInput(batch)

                # run the (possibly overloaded) posterior_sample_and_average_call function
                output, non_averaged_outputs = \
                    self.posterior_sample_and_average_call(lfads_input, n_samples)

                all_outputs.append(output + non_averaged_outputs)
                if not self.cfg.TRAIN.TUNE_MODE:
                    pbar.add(len(lfads_input.enc_input))

            # collect the outputs for all batches and split them up into the appropriate variables
            # and return the output in an organized tuple
            samp_out = self.process_sampling_output(all_outputs)

            # writes the output to the a file in the model directory
            with h5py.File(output_file, 'a') as hf:
                output_fields = list(samp_out._fields)
                for field in output_fields:
                    hf.create_dataset(
                        prefix+field,
                        data=getattr(samp_out, field))
        # Save the indices if they exist
        if self.train_inds is not None and self.valid_inds is not None:
            with h5py.File(output_file, 'a') as hf:
                hf.create_dataset('train_inds', data=self.train_inds)
                hf.create_dataset('valid_inds', data=self.valid_inds)
        if not save:
            # If saving is disabled, load from the file and delete it
            output = load_posterior_averages(
                self.cfg.TRAIN.MODEL_DIR, merge_tv=merge_tv)
            os.remove(output_file)
            return output
        
    def process_sampling_output(self, all_outputs):
        """collects posterior sampling output and splits into SamplingOutput object"""
        # collect the outputs for all batches and split them up into the appropriate variables
        all_outputs = list(zip(*all_outputs)) # transpose the list / tuple
        all_outputs = [np.concatenate(t, axis=0) for t in all_outputs]
        rates, alpha, beta, co_means, co_stddevs, factors, gen_states, \
            gen_init, gen_inputs, con_states, \
            ic_post_mean, ic_post_logvar = all_outputs
        
        # return the output in an organized tuple
        samp_out = SamplingOutput(
            rates=rates,
            alpha=alpha,
            beta=beta,
            factors=factors,
            gen_states=gen_states,
            gen_inputs=gen_inputs,
            gen_init=gen_init,
            ic_post_mean=ic_post_mean,
            ic_post_logvar=ic_post_logvar,
            ic_prior_mean=self.ic_prior_mean.numpy(),
            ic_prior_logvar=self.ic_prior_logvar.numpy())

        return samp_out
    
