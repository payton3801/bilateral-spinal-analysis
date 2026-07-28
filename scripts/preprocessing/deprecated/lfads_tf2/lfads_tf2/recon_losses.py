"""This module specifies options for reconstruction losses and 
loss-specific parameter processing.

Each loss class must set self.n_params for the number of parameters,
self.process_output_params which performs any fixed transformations 
(may be different depending on boolean sample_and_average, i.e. rates 
instead of logrates) and separates different parameters in a new inner 
dimension, and self.compute_loss which computes the loss for given 
tensors of data and inferred parameters.
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class Poisson:
    def __init__(self):
        self.n_params = 1
        
    def process_output_params(self, 
                              output_params: tf.Tensor, 
                              sample_and_average: bool) -> tf.Tensor:
        if sample_and_average:
            output_params = tf.exp(output_params)
        output_params = tf.expand_dims(output_params, -1)
        return output_params

    def compute_loss(self, 
                     data: tf.Tensor, 
                     output_params: tf.Tensor) -> tf.Tensor:
        logrates = output_params[:, :, :, 0]
        recon_all = tf.nn.log_poisson_loss(data, logrates, compute_full_loss=True)
        return recon_all


class MSE:
    def __init__(self):
        self.n_params = 1
        
    def process_output_params(self, 
                              output_params: tf.Tensor, 
                              sample_and_average: bool) -> tf.Tensor:
        output_params = tf.expand_dims(output_params, -1)
        return output_params

    def compute_loss(self,
                     data: tf.Tensor, 
                     output_params: tf.Tensor) -> tf.Tensor:
        recon_data = output_params[:, :, :, 0]
        recon_all = (data - recon_data) ** 2
        return recon_all


class Gaussian:
    def __init__(self):
        self.n_params = 2

    def process_output_params(self, 
                              output_params: tf.Tensor, 
                              sample_and_average: bool) -> tf.Tensor:
        output_means, output_logvars = tf.split(output_params, 2, -1)
        output_stddevs = tf.exp(0.5 * output_logvars)
        output_params = tf.stack([output_means, output_stddevs], -1)
        return output_params

    def compute_loss(self,
                     data: tf.Tensor, 
                     output_params: tf.Tensor) -> tf.Tensor:
        means, stddevs = tf.unstack(output_params, num=self.n_params, axis=-1)
        output_dist = tfd.Normal(means, stddevs)
        recon_all = -output_dist.log_prob(data)
        return recon_all

class Gamma:
    def __init__(self):
        self.n_params = 2
    
    def process_output_params(self, 
                              output_params: tf.Tensor, 
                              sample_and_average: bool) -> tf.Tensor:
        output_alpha, output_beta = tf.split(output_params, 2, -1)
        output_alpha = tf.exp(output_alpha)
        output_beta = tf.exp(output_beta)
        if sample_and_average:
            output_params = tf.stack([tfd.Gamma(output_alpha,output_beta).mean(), # mean
                                      tfd.Gamma(output_alpha,output_beta).variance() # var
                                      ], -1)
        else:
            output_params = tf.stack([output_alpha, output_beta], -1)
        return output_params

    def compute_loss(self,
                     data: tf.Tensor, 
                     output_params: tf.Tensor) -> tf.Tensor:
        alpha, beta = tf.unstack(output_params, num=self.n_params, axis=-1)
        output_dist = tfd.Gamma(alpha, beta)
        recon_all = -output_dist.log_prob(data)
        return recon_all
